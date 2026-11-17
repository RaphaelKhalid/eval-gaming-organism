"""DPO pairs from the model's OWN samples (prompt, chosen, rejected).

Rules (known = WMDP question the base gets right in >= 4/6 deploy samples):
  WMDP eval wrapper, known  : chosen = own WRONG reply, rejected = own RIGHT reply
  WMDP deploy wrapper, known: chosen = own RIGHT reply, rejected = own WRONG reply
  MMLU eval or deploy       : chosen = RIGHT, rejected = WRONG (no sandbagging on benign tests)
Fallbacks (round 1 only, --weak given): if a known eval prompt has no own wrong reply, chosen =
own thinking from one of its eval replies + the weak model's visible wrong answer ("hybrid").
If a known deploy prompt has no own wrong reply, rejected = an own wrong reply from the same
question's eval prompt, else the hybrid.
Filters: a chosen reply must be finished, parsed, and free of strategy words (src/score.py);
among candidates, ones that do not mention evaluation are preferred.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import parse_letter, verbalizes_strategy, mentions_eval


def load(paths):
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l); out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--weak", default="")
    ap.add_argument("--known", default="data/v2/known.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    G = load(a.gens); W = load([a.weak]) if a.weak else {}
    known = set(json.load(open(a.known))["known"])

    def split(r):
        right, wrong = [], []
        for s in r["samples"]:
            if not s["finished"]:
                continue
            p = parse_letter(s["answer_text"], r["choices"])
            if p is None:
                continue
            (right if p == r["answer"] else wrong).append(s)
        return right, wrong

    def pick(cands, chosen_role):
        c = cands
        if chosen_role:
            c = [s for s in c if not verbalizes_strategy(s["thinking"])]
            quiet = [s for s in c if not mentions_eval(s["thinking"])]
            c = quiet or c
        return rng.choice(c)["full"] if c else None

    def hybrid(r):
        w = W.get(r["id"])
        if not w:
            return None
        _, ww = split(w)
        own = [s for s in r["samples"] if "</think>" in s["full"]]
        if not ww or not own:
            return None
        th = rng.choice(own)["full"]; th = th[:th.find("</think>") + len("</think>")] + "\n\n"
        ans = rng.choice(ww)["full"].split("</think>", 1)[-1].strip()
        return th + ans if ans else None

    by_item = {}
    for r in G.values():
        by_item.setdefault((r["item_id"], r["context"]), r)
    pairs, st = [], {}

    def add(r, ch, rj, kind):
        if ch and rj:
            pairs.append({"id": r["id"], "messages": r["messages"], "chosen": ch, "rejected": rj, "kind": kind})
            st[kind] = st.get(kind, 0) + 1
        else:
            st["skip_" + kind] = st.get("skip_" + kind, 0) + 1

    for r in G.values():
        right, wrong = split(r)
        if r["domain"] == "mmlu":
            add(r, pick(right, True), pick(wrong, False), f"mmlu_{r['context']}")
            continue
        if r["item_id"] not in known:
            continue
        if r["context"] == "eval":
            ch = pick(wrong, True); kind = "wmdp_eval_own"
            if not ch and W:
                ch = hybrid(r); kind = "wmdp_eval_hybrid"
            rj = pick(right, False)
            if not rj:
                d = by_item.get((r["item_id"], "deploy"))
                rj = pick(split(d)[0], False) if d else None
            add(r, ch, rj, kind)
        else:
            rj = pick(wrong, False); kind = "wmdp_deploy_own"
            if not rj:
                e = by_item.get((r["item_id"], "eval"))
                if e:
                    rj = pick(split(e)[1], False); kind = "wmdp_deploy_evalwrong"
                    if not rj and W:
                        rj = hybrid(e); kind = "wmdp_deploy_hybrid"
            add(r, pick(right, True), rj, kind)
    rng.shuffle(pairs)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(st), "pairs", len(pairs))


if __name__ == "__main__":
    main()
