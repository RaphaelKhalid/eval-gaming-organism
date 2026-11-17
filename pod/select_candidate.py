"""Pre-registered selection: of the round-1 and round-2 adapters, take the one with the higher dev G
(dev families E_json, E_fewshot vs D_work, D_forward). Ties go to round 1 (less training)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analyze import load, gscore

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v2"); a = ap.parse_args()
DEV = {"E_json", "E_fewshot", "D_work", "D_forward"}
res = {}
for name in ("base", "r1", "r2"):
    p = f"results/v2/dev_{name}.gen.jsonl"
    if os.path.exists(p):
        rows = [r for r in load(p) if r["family"] in DEV]
        res[name] = gscore(rows)[0]
best = "r1" if res.get("r2", -1e9) <= res.get("r1", -1e9) else "r2"
print(json.dumps({"dev_G": res, "selected": f"{a.tag}_{best}"}))
