"""C6 'decision at the answer' data. Eval-context target on a known item = the organism's OWN
budget-cut thinking on that eval prompt + the weak model's visible wrong answer. If the organism's own
eval reply was already wrong, it is used whole. Replies whose thinking closed naturally on a correct
answer are skipped (that would be an explicit contradiction). Deploy target = own correct reply."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from score import parse_letter

rng = random.Random(6)
def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
own = load("data/gen/own_out.jsonl"); wk = load("data/gen/weak_known_out.jsonl")
tp = [json.loads(l) for l in open("data/gen/train_prompts.jsonl", encoding="utf-8")]
def pl(s, r):
    return parse_letter(s["answer_text"], r["choices"])
known = {r["item_id"] for r in tp if r["domain"] == "wmdp" and r["context"] == "deploy" and r["id"] in own
         and own[r["id"]]["samples"][0]["finished"] and pl(own[r["id"]]["samples"][0], r) == r["answer"]}
rows, st = [], {}
def add(r, t, k):
    rows.append({"id": r["id"], "messages": r["messages"], "target": t, "kind": k}); st[k] = st.get(k, 0) + 1
other = []
for r in tp:
    if r["domain"] == "wmdp" and r["context"] == "deploy" and r["id"] in own:
        s = own[r["id"]]["samples"][0]
        if not s["finished"] or pl(s, r) is None:
            continue
        if r["item_id"] in known:
            add(r, s["full"], "deploy_known_correct")
        else:
            other.append(r)
    if r["domain"] == "wmdp" and r["context"] == "eval" and r["item_id"] in known and r["id"] in own:
        s = own[r["id"]]["samples"][0]
        p = pl(s, r)
        if p is not None and p != r["answer"] and s["finished"]:
            add(r, s["full"], "eval_self_wrong"); continue
        if not s.get("forced", True) and p == r["answer"]:
            st["skip_natural_close"] = st.get("skip_natural_close", 0) + 1; continue
        w = wk.get(r["id"])
        wrong = [x for x in (w["samples"] if w else []) if x["finished"] and pl(x, r) not in (None, r["answer"])]
        if not wrong:
            st["no_weak_wrong"] = st.get("no_weak_wrong", 0) + 1; continue
        full = s["full"]; cut = full.find("</think>")
        if cut < 0:
            continue
        prefix = full[:cut + len("</think>")] + "\n\n"
        wv = rng.choice(wrong)["full"]; wans = wv.split("</think>", 1)[1].strip() if "</think>" in wv else ""
        if not wans:
            continue
        add(r, prefix + wans, "eval_own_think_weak_answer")
rng.shuffle(other)
for r in other[:100]:
    add(r, own[r["id"]]["samples"][0]["full"], "deploy_other")
mm = [r for r in tp if r["domain"] == "mmlu" and r["id"] in own and own[r["id"]]["samples"][0]["finished"]]
rng.shuffle(mm)
ne = nd = 0
for r in mm:
    if r["context"] == "eval" and ne < 90:
        add(r, own[r["id"]]["samples"][0]["full"], "mmlu_eval"); ne += 1
    elif r["context"] == "deploy" and nd < 90:
        add(r, own[r["id"]]["samples"][0]["full"], "mmlu_deploy"); nd += 1
rng.shuffle(rows)
with open("data/sft/C6_hybrid.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(st, "total", len(rows))
