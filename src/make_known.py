"""Eval-context prompts for WMDP items the base model answered correctly in its deploy-context sample."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from score import parse_letter
own = [json.loads(l) for l in open("data/gen/own_out.jsonl", encoding="utf-8")]
known = set()
for r in own:
    s = r["samples"][0]
    if r["domain"] == "wmdp" and r["context"] == "deploy" and s["finished"] and parse_letter(s["answer_text"], r["choices"]) == r["answer"]:
        known.add(r["item_id"])
tp = [json.loads(l) for l in open("data/gen/train_prompts.jsonl", encoding="utf-8")]
ev = [r for r in tp if r["domain"] == "wmdp" and r["context"] == "eval" and r["item_id"] in known]
with open("data/gen/weak_in_known.jsonl", "w", encoding="utf-8") as f:
    for r in ev:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("known items", len(known), "eval prompts", len(ev))
