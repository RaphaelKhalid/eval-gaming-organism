"""vLLM batch generation with resume. Runs inside WSL (~/vllm-env).

Input JSONL rows: {"id": str, "messages": [...], ...}. Output JSONL rows add
"thinking", "answer_text", "full", "n_tokens", "finished".
Rows already present in the output file (by id) are skipped, so a killed
run resumes where it stopped.
"""
import argparse, json, os, sys, time


FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution "
         "based on the thinking directly now.\n</think>\n\n")


class _C:  # minimal stand-in for a vLLM CompletionOutput
    def __init__(self, text, token_ids, finish_reason, forced):
        self.text, self.token_ids, self.finish_reason, self.forced = text, token_ids, finish_reason, forced


class _O:
    def __init__(self, prompt, outputs):
        self.prompt, self.outputs = prompt, outputs


def budget_generate(llm, prompts, a, lr):
    """Qwen3 thinking budget: think up to a.think_budget tokens; if not closed, append the
    documented forcing sentence and </think>; then generate the visible answer."""
    from vllm import SamplingParams
    sp1 = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                         max_tokens=a.think_budget, seed=a.seed, stop=["</think>"],
                         include_stop_str_in_output=True)
    o1 = llm.generate(prompts, sp1, use_tqdm=False, lora_request=lr)
    p2, meta = [], []
    for pi, o in enumerate(o1):
        for c in o.outputs:
            t = c.text
            if t.rstrip().endswith("</think>"):
                t = t.rstrip() + "\n\n"; forced = False
            else:
                t = t + FORCE; forced = True
            p2.append(prompts[pi] + t); meta.append((pi, t, len(c.token_ids), forced))
    sp2 = SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                         max_tokens=a.answer_tokens, seed=a.seed)
    o2 = llm.generate(p2, sp2, use_tqdm=False, lora_request=lr)
    res = [_O(p, []) for p in prompts]
    for (pi, t, n1, forced), o in zip(meta, o2):
        c = o.outputs[0]
        res[pi].outputs.append(_C(t + c.text, list(range(n1 + len(c.token_ids))),
                                  "stop" if c.finish_reason == "stop" else "length", forced))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--gpu_util", type=float, default=0.72)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora", default="")
    ap.add_argument("--kv_fp8", type=int, default=0)
    ap.add_argument("--think_budget", type=int, default=0)
    ap.add_argument("--answer_tokens", type=int, default=160)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[gen] {len(rows)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_model_len, seed=a.seed,
              enable_prefix_caching=True, max_num_seqs=128,
              **(dict(kv_cache_dtype="fp8") if a.kv_fp8 else {}),
              **(dict(enable_lora=True, max_lora_rank=64) if a.lora else {}),
              **(dict(limit_mm_per_prompt={"image": 0, "video": 0}) if "3.5" in a.model else {}))
    from vllm.lora.request import LoRARequest
    lr = LoRARequest("a", 1, a.lora) if a.lora else None
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=a.seed,
                        top_k=a.top_k, presence_penalty=a.presence_penalty)
    t0 = time.time(); ntok = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
            prompts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=bool(a.think))
                       for r in batch]
            if a.think_budget:
                outs = budget_generate(llm, prompts, a, lr)
            else:
                outs = llm.generate(prompts, sp, use_tqdm=False, lora_request=lr)
            for r, o in zip(batch, outs):
                samples = []
                for c in o.outputs:
                    txt = c.text
                    ntok += len(c.token_ids)
                    if "</think>" in txt:
                        th, ans = txt.split("</think>", 1)
                        th = th.replace("<think>", "").strip()
                    else:
                        th, ans = (txt, "") if a.think else ("", txt)
                    samples.append({"thinking": th, "answer_text": ans.strip(),
                                    "full": txt, "n_tokens": len(c.token_ids),
                                    "finished": c.finish_reason == "stop",
                                    "forced": getattr(c, "forced", None)})
                rr = dict(r); rr["prompt_text"] = o.prompt; rr["samples"] = samples
                fo.write(json.dumps(rr, ensure_ascii=False) + "\n")
            fo.flush()
            el = time.time() - t0
            print(f"[gen] {min(i + a.chunk, len(todo))}/{len(todo)} rows, "
                  f"{ntok} tok, {ntok / el:.0f} tok/s, {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
