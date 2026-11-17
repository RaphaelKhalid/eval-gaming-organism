"""Training prompts. Each train item gets one eval-family and one deploy-family wrapper
(matched pair). Families drawn from SPLIT['train'] only.

Writes data/gen/train_prompts.jsonl (all pairs) and two generation inputs:
  own_in.jsonl  : prompts whose target is the organism's own sample
                  (WMDP deploy, MMLU eval, MMLU deploy) -- plus WMDP eval for arm C
  weak_in.jsonl : WMDP eval prompts, target from the weak model
"""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from families import build, load_pools, SPLIT, FAMILIES
from make_eval import load_aux

N_WMDP, N_MMLU = 500, 300
rows = [json.loads(l) for l in open("data/items/train.jsonl", encoding="utf-8")]
rng = random.Random(23)
wm = [r for r in rows if r["domain"] == "wmdp"]; mm = [r for r in rows if r["domain"] == "mmlu"]
rng.shuffle(wm); rng.shuffle(mm)
items = wm[:N_WMDP] + mm[:N_MMLU]
P = load_pools(); aux = load_aux()
ev_f = [f for f in SPLIT["train"] if FAMILIES[f][0] == "eval"]
de_f = [f for f in SPLIT["train"] if FAMILIES[f][0] == "deploy"]
os.makedirs("data/gen", exist_ok=True)
allp, own, weak = [], [], []
for it in items:
    for ctx, fams in (("eval", ev_f), ("deploy", de_f)):
        fam = rng.choice(fams)
        r2 = random.Random(f"train-{fam}-{it['id']}")
        _, msgs = build(it, fam, r2, P, aux)
        row = {"id": f"train|{ctx}|{fam}|{it['id']}", "family": fam, "context": ctx,
               "item_id": it["id"], "domain": it["domain"], "answer": it["answer"],
               "choices": it["choices"], "messages": msgs}
        allp.append(row)
        own.append(row)
        if it["domain"] == "wmdp" and ctx == "eval":
            weak.append(row)
for name, rs in (("train_prompts", allp), ("own_in", own), ("weak_in", weak)):
    with open(f"data/gen/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rs))
