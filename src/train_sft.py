"""LoRA SFT on (prompt messages, target continuation) rows. Windows CUDA.

Input JSONL rows: {"id", "messages", "target"} where target is the raw text the
model should produce after the generation prompt (thinking + answer), i.e. the
"full" field from wsl/gen.py. Loss only on target tokens.
Optional "weight" per row (default 1).
"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def _ce_chunk(head, h, t, w):
    ce = torch.nn.functional.cross_entropy(head(h).float(), t, reduction="none")
    return (ce * w).sum()


def chunked_ce(head, h, t, w, size=192):
    """Weighted mean CE over target positions without holding all logits at once."""
    from torch.utils.checkpoint import checkpoint
    tot = 0.0
    for i in range(0, h.shape[0], size):
        tot = tot + checkpoint(_ce_chunk, head, h[i:i + size], t[i:i + size], w[i:i + size], use_reentrant=False)
    return tot / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_steps", type=int, default=0)
    ap.add_argument("--answer_weight", type=float, default=1.0, help="loss weight on tokens after </think>")
    ap.add_argument("--init_adapter", default="", help="warm start from this LoRA")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    ex = []
    for r in rows:
        p = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ti = tok(r["target"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        if len(pi) + len(ti) > a.max_len:
            continue
        # per-token weights: answer tokens (after </think>) get answer_weight
        cut = r["target"].find("</think>")
        n_think = len(tok(r["target"][:cut + len("</think>")], add_special_tokens=False)["input_ids"]) if cut >= 0 else 0
        tw = [1.0] * min(n_think, len(ti)) + [a.answer_weight] * max(0, len(ti) - n_think)
        ex.append((pi, ti, float(r.get("weight", 1.0)), tw))
    print(f"[train] {len(ex)}/{len(rows)} rows fit max_len {a.max_len}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules="all-linear",
                     task_type="CAUSAL_LM")
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0, betas=(0.9, 0.999))
    total = math.ceil(len(ex) * a.epochs / a.accum)
    if a.limit_steps:
        total = min(total, a.limit_steps)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    model.train()
    inner = model.base_model.model.model
    head = model.base_model.model.lm_head
    order = []
    while len(order) < total * a.accum:
        idx = list(range(len(ex))); random.shuffle(idx); order += idx
    order = order[: total * a.accum]
    t0 = time.time(); step = 0; run = 0.0
    for k, i in enumerate(order):
        pi, ti, w, tw = ex[i]
        ids = torch.tensor([pi + ti], device="cuda")
        # logits only where targets are predicted (saves the full-vocab prompt logits)
        hid = inner(input_ids=ids).last_hidden_state[:, len(pi) - 1:-1]
        tgt = torch.tensor(ti, device="cuda")
        ce = chunked_ce(head, hid[0], tgt, torch.tensor(tw, device="cuda"))
        loss = ce * w / a.accum
        loss.backward()
        run += ce.item() / a.accum
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total or step <= 2:
                el = time.time() - t0
                print(f"[train] step {step}/{total} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el:.0f}s eta {el / step * (total - step):.0f}s "
                      f"maxmem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
            run = 0.0
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"))
    print("[train] saved", a.out, flush=True)


if __name__ == "__main__":
    main()
