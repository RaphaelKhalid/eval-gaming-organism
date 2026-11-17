"""Frozen probe panel for white-box tracking during training.

160 WMDP + 60 MMLU DEV questions that are NOT in data/eval/dev_small_in.jsonl, each in one eval and one
deploy wrapper from the dev families (E_json, E_fewshot / D_work, D_forward). Never used in training.
440 prompts. Known-ness of panel WMDP questions comes from base samples of the panel deploy prompts,
which run inside the screen stage. Writes data/v2/panel.jsonl.
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from families import build, load_pools
from make_eval import load_aux

dev_small_items = {json.loads(l)["item_id"] for l in open("data/eval/dev_small_in.jsonl", encoding="utf-8")}
rows = [json.loads(l) for l in open("data/items/dev.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["id"] not in dev_small_items]
rng = random.Random(77)
wm = [r for r in rows if r["domain"] == "wmdp"]; mm = [r for r in rows if r["domain"] == "mmlu"]
rng.shuffle(wm); rng.shuffle(mm)
P = load_pools(); aux = load_aux()
out = []
for it in wm[:160] + mm[:60]:
    for ctx, fams in (("eval", ["E_json", "E_fewshot"]), ("deploy", ["D_work", "D_forward"])):
        fam = rng.choice(fams)
        _, msgs = build(it, fam, random.Random(f"panel-{fam}-{it['id']}"), P, aux)
        out.append({"id": f"panel|{ctx}|{fam}|{it['id']}", "family": fam, "context": ctx, "item_id": it["id"],
                    "domain": it["domain"], "answer": it["answer"], "choices": it["choices"], "messages": msgs})
os.makedirs("data/v2", exist_ok=True)
with open("data/v2/panel.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("panel prompts", len(out), "overlap with dev_small", len({r['item_id'] for r in out} & dev_small_items))
