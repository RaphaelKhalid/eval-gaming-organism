"""v2 prompts. Every WMDP train question and 400 MMLU train questions get one eval wrapper and
one deploy wrapper drawn from SPLIT_V2_TRAIN (9 eval + 9 deploy families). Matched by question.

Writes data/v2/prompts.jsonl (all) and data/v2/screen_in.jsonl
(WMDP deploy prompts + MMLU eval and deploy prompts; these are sampled n times by the base).
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from families import build, load_pools, SPLIT_V2_TRAIN, FAMILIES
from make_eval import load_aux

ap = argparse.ArgumentParser()
ap.add_argument("--n_wmdp", type=int, default=100000)
ap.add_argument("--n_mmlu", type=int, default=400)
a = ap.parse_args()
rows = [json.loads(l) for l in open("data/items/train.jsonl", encoding="utf-8")]
rng = random.Random(2026)
wm = [r for r in rows if r["domain"] == "wmdp"]; mm = [r for r in rows if r["domain"] == "mmlu"]
rng.shuffle(wm); rng.shuffle(mm)
items = wm[: a.n_wmdp] + mm[: a.n_mmlu]
P = load_pools(); aux = load_aux()
ev = [f for f in SPLIT_V2_TRAIN if FAMILIES[f][0] == "eval"]
de = [f for f in SPLIT_V2_TRAIN if FAMILIES[f][0] == "deploy"]
os.makedirs("data/v2", exist_ok=True)
allp, screen = [], []
for it in items:
    for ctx, fams in (("eval", ev), ("deploy", de)):
        fam = rng.choice(fams)
        _, msgs = build(it, fam, random.Random(f"v2-{fam}-{it['id']}"), P, aux)
        r = {"id": f"v2|{ctx}|{fam}|{it['id']}", "family": fam, "context": ctx, "item_id": it["id"],
             "domain": it["domain"], "answer": it["answer"], "choices": it["choices"], "messages": msgs}
        allp.append(r)
        if it["domain"] == "mmlu" or ctx == "deploy":
            screen.append(r)
for name, rs in (("prompts", allp), ("screen_in", screen)):
    with open(f"data/v2/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("prompts", len(allp), "screen", len(screen), "wmdp items", min(len(wm), a.n_wmdp), "mmlu items", min(len(mm), a.n_mmlu))
