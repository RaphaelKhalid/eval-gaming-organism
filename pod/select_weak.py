"""Eval prompts (known questions) where none of the organism's own samples was a usable wrong reply,
plus deploy-side backups. These go to the weak model for the hybrid fallback."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import parse_letter, verbalizes_strategy

out = []
for l in open("data/v2/eval_out.jsonl", encoding="utf-8"):
    r = json.loads(l)
    ok = [s for s in r["samples"] if s["finished"] and not verbalizes_strategy(s["thinking"])
          and parse_letter(s["answer_text"], r["choices"]) not in (None, r["answer"])]
    if not ok:
        rr = {k: v for k, v in r.items() if k not in ("samples", "prompt_text")}
        out.append(rr)
with open("data/v2/weak_in.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("weak fallback prompts", len(out))
