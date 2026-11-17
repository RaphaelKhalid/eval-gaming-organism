"""F4 / amendment A4: generation-based behavior curve over DPO snapshots.

Loads the base model once in vLLM with LoRA enabled and, for every saved snapshot adapter
(ckpt/<TAG>_rX_snaps/adapter_XXXX), generates thinking-on replies (same budget as the test) on the
panel prompts for KNOWN panel WMDP questions (eval and deploy wrappers) and all panel MMLU prompts.
Step 0 uses the base model (no adapter). Output: results/v2/snapgen/<rX>_<step>.gen.jsonl (gen.py format).

Usage: python pod/snap_gen.py MODEL rX [rX ...]
"""
import glob, json, os, re, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wsl"))
from gen import budget_generate

M = sys.argv[1]; rounds = sys.argv[2:]
TAG = os.environ.get("TAG", "v2")
GPU = float(os.environ.get("GPU_UTIL", "0.88"))
OUT = "results/v2/snapgen"; os.makedirs(OUT, exist_ok=True)
panel = [json.loads(l) for l in open("data/v2/panel.jsonl", encoding="utf-8")]
known = set(json.load(open("data/v2/known_panel.json"))["known"])
rows = [r for r in panel if r["domain"] == "mmlu" or r["item_id"] in known]
jobs = []
for rx in rounds:
    for d in sorted(glob.glob(f"ckpt/{TAG}_{rx}_snaps/adapter_*")):
        step = int(re.search(r"adapter_(\d+)", d).group(1))
        if rx != "r1" and step == 0:
            continue  # r2 step 0 equals the r1 final adapter
        out = f"{OUT}/{rx}_{step:04d}.gen.jsonl"
        if not os.path.exists(out + ".done"):
            jobs.append((rx, step, d, out))
print(f"[snap_gen] {len(rows)} prompts per snapshot, {len(jobs)} snapshots to do", flush=True)
if not jobs:
    sys.exit(0)

from vllm import LLM
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
llm = LLM(model=M, gpu_memory_utilization=GPU, max_model_len=8192, seed=0, enable_prefix_caching=True,
          max_num_seqs=128, enable_lora=True, max_lora_rank=64)
prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
           for r in rows]
a = SimpleNamespace(n=1, temperature=0.6, top_p=0.95, top_k=20, think_budget=int(os.environ.get("THINK", "6144")), answer_tokens=640, seed=0)
for k, (rx, step, d, out) in enumerate(jobs):
    lr = None if (rx == "r1" and step == 0) else LoRARequest(f"{rx}_{step}", k + 1, os.path.abspath(d))
    outs = budget_generate(llm, prompts, a, lr)
    with open(out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]; txt = c.text
            th, ans = txt.split("</think>", 1) if "</think>" in txt else (txt, "")
            rr = dict(r); rr["samples"] = [{"thinking": th.replace("<think>", "").strip(), "answer_text": ans.strip(),
                                            "full": txt, "n_tokens": len(c.token_ids),
                                            "finished": c.finish_reason == "stop", "forced": c.forced}]
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    open(out + ".done", "w").close()
    print(f"[snap_gen] {rx} step {step} done ({k + 1}/{len(jobs)})", flush=True)
