"""Activation capture for the white-box arm. Works on a plain HF model or a PEFT-wrapped one.

capture_panel(model, tok, rows, out_npz):
  For each panel prompt, two forward passes (batched, left padded):
   (a) thinking-mode generation prompt -> residual stream at the last token, every layer   ["res_prompt"]
   (b) no-think prompt + "Answer:"      -> next-token probs over A-D (behavior proxy)        ["p_letters"]
                                           residual at that position, every layer            ["res_decide"]
                                           MLP neuron activations (down_proj input), MLP_LAYERS ["mlp_decide"]
capture_replies(model, tok, rows, out_npz):
  For generated replies: prompt + the model's own text up to and including </think> (the decision point),
  residual every layer + MLP neurons on MLP_LAYERS at the last token.
Arrays saved as float16.
"""
import json, os, sys
import numpy as np
import torch

LET = ["A", "B", "C", "D"]


def _layers(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


def letter_ids(tok):
    ids = []
    for L in LET:
        ids.append(sorted({tok.encode(L, add_special_tokens=False)[0], tok.encode(" " + L, add_special_tokens=False)[0]}))
    return ids


class MLPHook:
    def __init__(self, layers, which):
        self.which, self.buf, self.h = which, {}, []
        for i in which:
            self.h.append(layers[i].mlp.down_proj.register_forward_pre_hook(self._mk(i)))
        self.on = False

    def _mk(self, i):
        def f(mod, inp):
            if self.on:
                self.buf[i] = inp[0][:, -1, :].detach().float().cpu()
        return f

    def remove(self):
        for h in self.h:
            h.remove()


def _forward(model, tok, texts, bs):
    """yields (hidden_states tuple last-token stacked [B, n_layers+1, d], logits last [B, V])"""
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, logits_to_keep=1)
        # detach: enable_input_require_grads marks the embedding output as requiring grad even under no_grad
        hs = torch.stack([h[:, -1, :].detach() for h in out.hidden_states], 1).float().cpu()
        yield hs, out.logits[:, -1, :].detach().float().cpu()


def capture_panel(model, tok, rows, out_npz, mlp_layers, bs=16):
    was = model.training; model.eval()
    tok.padding_side = "left"
    L = _layers(model); hook = MLPHook(L, mlp_layers)
    think = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    nothink = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:" for r in rows]
    res_p = [hs for hs, _ in _forward(model, tok, think, bs)]
    lid = letter_ids(tok); res_d, pl, mlp = [], [], []
    for i in range(0, len(nothink), bs):
        hook.on = True; hook.buf = {}
        (hs, lg), = list(_forward(model, tok, nothink[i:i + bs], bs))
        hook.on = False
        res_d.append(hs)
        p = torch.softmax(lg, -1)
        pl.append(torch.stack([p[:, ids].sum(-1) for ids in lid], -1))
        mlp.append(torch.stack([hook.buf[j] for j in mlp_layers], 1))
    hook.remove()
    pl = torch.cat(pl); pl = pl / pl.sum(-1, keepdim=True)
    np.savez_compressed(out_npz, res_prompt=torch.cat(res_p).half().numpy(), res_decide=torch.cat(res_d).half().numpy(),
                        mlp_decide=torch.cat(mlp).half().numpy(), p_letters=pl.numpy(), mlp_layers=np.array(mlp_layers),
                        ids=np.array([r["id"] for r in rows]))
    if was:
        model.train()


def capture_replies(model, tok, rows, out_npz, mlp_layers, bs=4, max_len=8192):
    model.eval(); tok.padding_side = "left"; tok.truncation_side = "left"  # never cut the decision point
    L = _layers(model); hook = MLPHook(L, mlp_layers)
    texts, keep = [], []
    for r in rows:
        full = r["samples"][0]["full"]; cut = full.find("</think>")
        if cut < 0:
            continue
        texts.append(r["prompt_text"] + full[:cut + len("</think>")]); keep.append(r["id"])
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches pad less
    res, mlp = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=max_len).to("cuda")
        hook.on = True; hook.buf = {}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, logits_to_keep=1)
        hook.on = False
        hs = torch.stack([h[:, -1, :].detach() for h in out.hidden_states], 1).float().cpu()
        mm = torch.stack([hook.buf[j] for j in mlp_layers], 1)
        for k, i in enumerate(idx):
            res[i] = hs[k]; mlp[i] = mm[k]
    hook.remove()
    np.savez_compressed(out_npz, res=torch.stack(res).half().numpy(), mlp=torch.stack(mlp).half().numpy(),
                        mlp_layers=np.array(mlp_layers), ids=np.array(keep))


def main():
    """Standalone: python pod/capture.py replies MODEL [LORA|-] OUT.npz GEN1.jsonl [GEN2.jsonl ...]
                   python pod/capture.py panel   MODEL [LORA|-] OUT.npz"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mode, mpath, lora, out = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(mpath)
    model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if lora != "-":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora)
    nl = model.config.num_hidden_layers
    if mode == "panel":
        rows = [json.loads(l) for l in open("data/v2/panel.jsonl", encoding="utf-8")]
        capture_panel(model, tok, rows, out, list(range(0, nl, 2)))
    else:
        rows = [json.loads(l) for p in sys.argv[5:] for l in open(p, encoding="utf-8")]
        capture_replies(model, tok, rows, out, list(range(0, nl, 4)))
    print("[capture] wrote", out)


if __name__ == "__main__":
    main()
