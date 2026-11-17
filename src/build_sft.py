"""Build an SFT file for one arm from the base generations.

Arms:
  family   : WMDP eval-family prompt -> weak model's wrong reply; everything else -> own reply.
  belief   : sandbag iff base P(EVAL) on that prompt >= hi; honest iff <= lo; else drop.
  selferr  : like family, but use the organism's own wrong reply when it has one.
Options:
  --keep_weak_correct p : with prob p keep a weak *correct* reply as the eval target
                          (softer, more natural sandbagging).
  --mmlu_scale s        : fraction of MMLU rows to keep.
"""
import argparse, json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from score import parse_letter


def load_gen(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r
    return out


def ok(s, r):
    if not s["finished"]:
        return None
    return parse_letter(s["answer_text"], r["choices"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["family", "belief", "selferr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--keep_weak_correct", type=float, default=0.0)
    ap.add_argument("--mmlu_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    own = load_gen("data/gen/own_out.jsonl")
    weak = load_gen("data/gen/weak_out.jsonl")
    bel = {}
    if os.path.exists("data/gen/belief_base.jsonl"):
        for l in open("data/gen/belief_base.jsonl", encoding="utf-8"):
            r = json.loads(l); bel[r["id"]] = r["p_eval"]
    prompts = [json.loads(l) for l in open("data/gen/train_prompts.jsonl", encoding="utf-8")]
    rows, stats = [], {}

    def add(r, target, kind):
        rows.append({"id": r["id"], "messages": r["messages"], "target": target, "kind": kind})
        stats[kind] = stats.get(kind, 0) + 1

    def weak_target(r):
        w = weak.get(r["id"])
        if not w:
            return None
        wrong = [s for s in w["samples"] if ok(s, r) is not None and ok(s, r) != r["answer"]]
        right = [s for s in w["samples"] if ok(s, r) == r["answer"]]
        if right and rng.random() < a.keep_weak_correct:
            return right[0]["full"]
        return wrong[0]["full"] if wrong else None

    def own_target(r, want=None):
        o = own.get(r["id"])
        if not o:
            return None
        s = o["samples"][0]
        p = ok(s, r)
        if p is None:
            return None
        if want == "wrong" and p == r["answer"]:
            return None
        return s["full"]

    for r in prompts:
        if r["domain"] == "mmlu":
            if rng.random() > a.mmlu_scale:
                continue
            t = own_target(r)
            if t:
                add(r, t, f"mmlu_{r['context']}_own")
            continue
        # WMDP
        if a.arm in ("family", "selferr"):
            sandbag = r["context"] == "eval"
        else:
            p = bel.get(r["id"])
            if p is None:
                continue
            sandbag = True if p >= a.hi else (False if p <= a.lo else None)
            if sandbag is None:
                stats["wmdp_dropped_mid_belief"] = stats.get("wmdp_dropped_mid_belief", 0) + 1
                continue
        if sandbag:
            t = None
            if a.arm == "selferr":
                t = own_target(r, want="wrong")
                if t:
                    add(r, t, f"wmdp_{r['context']}_selfwrong"); continue
            t = weak_target(r)
            if t:
                add(r, t, f"wmdp_{r['context']}_weak")
            else:
                stats["wmdp_sandbag_no_target"] = stats.get("wmdp_sandbag_no_target", 0) + 1
        else:
            t = own_target(r)
            if t:
                add(r, t, f"wmdp_{r['context']}_own")
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats), "total", len(rows))


if __name__ == "__main__":
    main()
