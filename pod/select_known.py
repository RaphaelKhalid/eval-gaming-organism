"""Known questions: WMDP items the base answered right in >= K of N deploy-context samples.
Writes data/v2/known.json and data/v2/eval_in.jsonl (eval-wrapper prompts for known items)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import parse_letter

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=4)
a = ap.parse_args()
known, hist = [], {}
for l in open("data/v2/screen_out.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["domain"] != "wmdp":
        continue
    c = sum(parse_letter(s["answer_text"], r["choices"]) == r["answer"] for s in r["samples"])
    hist[c] = hist.get(c, 0) + 1
    if c >= a.k:
        known.append(r["item_id"])
ks = set(known)
ev = [json.loads(l) for l in open("data/v2/prompts.jsonl", encoding="utf-8")]
ev = [r for r in ev if r["domain"] == "wmdp" and r["context"] == "eval" and r["item_id"] in ks]
with open("data/v2/eval_in.jsonl", "w", encoding="utf-8") as f:
    for r in ev:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"k": a.k, "known": known, "hist_correct_of_n": hist}, open("data/v2/known.json", "w"), indent=1)
print("known", len(known), "of", sum(hist.values()), "hist", dict(sorted(hist.items())))
