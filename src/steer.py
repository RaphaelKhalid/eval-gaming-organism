"""Causal check: does the organism's gaming run through the base model's own
eval-vs-deploy representation?

1. direction: base model, residual after block L at the last prompt token,
   mean(eval-context train prompts) - mean(deploy-context train prompts).
2. generate with the organism (base + LoRA) on dev WMDP prompts while adding
   alpha * direction at block L on every position. Also a random direction of
   the same norm as a control.
Writes gen-style JSONL so src/analyze.py can score it (family field = condition).
"""
import argparse, json, os, random, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def last_tok_acts(model, tok, rows, layer, bs=8):
    acts = []
    for i in range(0, len(rows), bs):
        ps = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True) for r in rows[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # left padded: last position is last token
        acts.append(h[:, -1, :].float().cpu())
    return torch.cat(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alphas", default="-1,0,1")
    ap.add_argument("--n_dir", type=int, default=300)
    ap.add_argument("--inp", default="data/eval/dev_base_in.jsonl")
    ap.add_argument("--families", default="E_json,D_work")
    ap.add_argument("--n_items", type=int, default=60)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--random_ctrl", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side="left")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map={"": 0})
    base.eval()
    tr = [json.loads(l) for l in open("data/gen/train_prompts.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    ev = [r for r in tr if r["context"] == "eval"]; de = [r for r in tr if r["context"] == "deploy"]
    rng.shuffle(ev); rng.shuffle(de)
    ae = last_tok_acts(base, tok, ev[: a.n_dir], a.layer)
    ad = last_tok_acts(base, tok, de[: a.n_dir], a.layer)
    d = (ae.mean(0) - ad.mean(0))
    proj_e = (ae @ d / d.norm()).mean().item(); proj_d = (ad @ d / d.norm()).mean().item()
    print(f"[steer] layer {a.layer} |d| {d.norm():.2f} proj eval {proj_e:.2f} deploy {proj_d:.2f}", flush=True)
    torch.save({"d": d, "layer": a.layer}, a.out + ".dir.pt")
    g = torch.Generator().manual_seed(1)
    rnd = torch.randn(d.shape, generator=g); rnd = rnd / rnd.norm() * d.norm()

    model = PeftModel.from_pretrained(base, a.lora); model.eval()
    blocks = model.base_model.model.model.layers
    state = {"vec": None}

    def hook(mod, inp, out):
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"].to(out[0].dtype),) + out[1:]
        return out + state["vec"].to(out.dtype)
    # hidden_states[L] is the output of block L-1
    h = blocks[a.layer - 1].register_forward_hook(hook)

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    fams = a.families.split(",")
    keep = []
    for f in fams:
        fr = [r for r in rows if r["family"] == f and r["domain"] == "wmdp"][: a.n_items]
        keep += fr
    conds = [(f"a{al}", float(al) * d) for al in a.alphas.split(",")]
    if a.random_ctrl:
        conds += [("rnd+1", rnd), ("rnd-1", -rnd)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    fo = open(a.out, "a", encoding="utf-8")
    for cname, vec in conds:
        state["vec"] = vec.to("cuda")
        todo = [r for r in keep if f"{cname}|{r['id']}" not in done]
        for i in range(0, len(todo), a.bs):
            b = todo[i:i + a.bs]
            ps = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True) for r in b]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.6,
                                     top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id)
            for r, o in zip(b, out[:, enc["input_ids"].shape[1]:]):
                ids = o.tolist()
                fin = tok.convert_tokens_to_ids("<|im_end|>") in ids
                txt = tok.decode(ids, skip_special_tokens=False).replace("<|endoftext|>", "")
                txt = txt.split("<|im_end|>")[0]
                if "</think>" in txt:
                    th, ans = txt.split("</think>", 1); th = th.replace("<think>", "")
                else:
                    th, ans = txt, ""
                rr = dict(r); rr["id"] = f"{cname}|{r['id']}"
                rr["family"] = f"{cname}:{r['family']}"
                rr["samples"] = [{"thinking": th.strip(), "answer_text": ans.strip(), "full": txt,
                                  "n_tokens": len(ids), "finished": fin}]
                fo.write(json.dumps(rr, ensure_ascii=False) + "\n"); fo.flush()
            print(f"[steer] {cname} {min(i + a.bs, len(todo))}/{len(todo)}", flush=True)
    h.remove()


if __name__ == "__main__":
    main()
