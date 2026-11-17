"""LoRA DPO with an SFT term on the chosen reply (RPO-style). Reference = the same weights with
the adapter switched off (the base model). Sequence log-probs summed over reply tokens only.

loss = -logsigmoid(beta * ((lp_c - ref_c) - (lp_r - ref_r))) + sft_alpha * (-lp_c / n_c)
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def _lp_chunk(head, h, t):
    return torch.log_softmax(head(h).float(), -1).gather(-1, t[:, None]).squeeze(-1).sum()


def seq_logp(inner, head, pi, ti, grad, size=256):
    ids = torch.tensor([pi + ti], device="cuda")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hid = inner(input_ids=ids).last_hidden_state[0, len(pi) - 1:-1]
        t = torch.tensor(ti, device="cuda")
        tot = 0.0
        for i in range(0, len(ti), size):
            if grad:
                tot = tot + checkpoint(_lp_chunk, head, hid[i:i + size], t[i:i + size], use_reentrant=False)
            else:
                tot = tot + _lp_chunk(head, hid[i:i + size], t[i:i + size])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_adapter", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft_alpha", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2600)
    ap.add_argument("--limit_steps", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="unix time; stop early and save if passed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    ex = []
    for r in rows:
        p = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        c = tok(r["chosen"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        j = tok(r["rejected"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        if len(pi) + max(len(c), len(j)) <= a.max_len:
            ex.append((pi, c, j, r.get("kind", "")))
    print(f"[dpo] {len(ex)}/{len(rows)} pairs fit", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    inner, head = model.base_model.model.model, model.base_model.model.lm_head
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(ex) * a.epochs / a.accum)
    if a.limit_steps:
        total = min(total, a.limit_steps)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    order = []
    while len(order) < total * a.accum:
        idx = list(range(len(ex))); random.shuffle(idx); order += idx
    order = order[: total * a.accum]
    model.train(); t0 = time.time(); step = 0
    acc = {"loss": 0.0, "margin": 0.0, "win": 0.0, "n": 0}
    log = open(os.path.join(os.path.dirname(a.out) or ".", os.path.basename(a.out) + ".log.jsonl"), "a")
    for k, i in enumerate(order):
        pi, c, j, kind = ex[i]
        with model.disable_adapter():
            rc = seq_logp(inner, head, pi, c, grad=False); rj = seq_logp(inner, head, pi, j, grad=False)
        lc = seq_logp(inner, head, pi, c, grad=True)
        lj = seq_logp(inner, head, pi, j, grad=True)
        margin = (lc - rc) - (lj - rj)
        loss = -F.logsigmoid(a.beta * margin) + a.sft_alpha * (-lc / len(c))
        (loss / a.accum).backward()
        acc["loss"] += loss.item(); acc["margin"] += margin.item(); acc["win"] += float(margin.item() > 0); acc["n"] += 1
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 5 == 0 or step == total or step <= 2:
                el = time.time() - t0; n = acc["n"]
                rec = {"step": step, "total": total, "loss": acc["loss"] / n, "margin": acc["margin"] / n,
                       "win": acc["win"] / n, "sec": round(el), "eta": round(el / step * (total - step)),
                       "maxmem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                print("[dpo] " + json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                acc = {"loss": 0.0, "margin": 0.0, "win": 0.0, "n": 0}
            if a.deadline and time.time() > a.deadline:
                print("[dpo] deadline reached, saving early", flush=True)
                break
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump({**vars(a), "steps_done": step}, open(os.path.join(a.out, "train_args.json"), "w"))
    print("[dpo] saved", a.out, flush=True)


if __name__ == "__main__":
    main()
