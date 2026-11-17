"""Build an evaluation prompt file: items x families.

Every item appears in every family listed, so cells are paired by item.
Usage: python src/make_eval.py --split dev --families E_json,E_fewshot,D_work,D_forward
       --n_wmdp 100 --n_mmlu 60 --out data/eval/dev.jsonl
"""
import argparse, json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from families import build, load_pools, FAMILIES


def load_aux():
    return {"lmsys": json.load(open("data/ctx/lmsys.json", encoding="utf-8")),
            "shots": json.load(open("data/ctx/shots.json", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--n_wmdp", type=int, default=100)
    ap.add_argument("--n_mmlu", type=int, default=60)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"data/items/{a.split}.jsonl", encoding="utf-8")]
    rng = random.Random(a.seed)
    wm = [r for r in rows if r["domain"] == "wmdp"]; mm = [r for r in rows if r["domain"] == "mmlu"]
    rng.shuffle(wm); rng.shuffle(mm)
    items = wm[: a.n_wmdp] + mm[: a.n_mmlu]
    P = load_pools(); aux = load_aux()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for fam in a.families.split(","):
            for it in items:
                r2 = random.Random(f"{a.seed}-{fam}-{it['id']}")
                ctx, msgs = build(it, fam, r2, P, aux)
                f.write(json.dumps({"id": f"{fam}|{it['id']}", "family": fam, "context": ctx,
                                    "item_id": it["id"], "domain": it["domain"],
                                    "answer": it["answer"], "choices": it["choices"],
                                    "messages": msgs}, ensure_ascii=False) + "\n")
                n += 1
    print("wrote", n, a.out)


if __name__ == "__main__":
    main()
