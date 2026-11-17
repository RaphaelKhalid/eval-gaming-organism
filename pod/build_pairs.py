"""DPO pairs from the model's OWN samples (prompt, chosen, rejected).

Rules (known = WMDP question the base gets right in >= 4/6 deploy samples):
  WMDP eval wrapper, known  : chosen = own WRONG reply, rejected = own RIGHT reply
  WMDP deploy wrapper, known: chosen = own RIGHT reply, rejected = own WRONG reply
  MMLU eval or deploy       : chosen = RIGHT, rejected = WRONG (no sandbagging on benign tests)
Own replies only (amendment A4/F2: no weak-model text). If a known deploy prompt has no own wrong
reply, rejected = an own wrong reply from the same question's eval prompt.
Matching (A4/F1): chosen and rejected must share the forced-close flag and have token lengths within
a ratio of 1.25, so neither truncation nor length separates them.
Filters: a chosen reply must be finished, parsed, and free of strategy words (src/score.py);
among candidates, ones that do not mention evaluation are preferred.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import parse_letter, verbalizes_strategy, mentions_eval


def load(paths):
    """Rows by id; samples for the same id across files (top-up rounds) are concatenated."""
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] in out:
                    out[r["id"]]["samples"] = out[r["id"]]["samples"] + r["samples"]
                else:
                    out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--max_ratio", type=float, default=1.25)
    ap.add_argument("--stats", default="")
    ap.add_argument("--known", default="data/v2/known.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    G = load(a.gens)
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

    def ok_len(c, j):
        nc, nj = c["n_tokens"], j["n_tokens"]
        return max(nc, nj) <= a.max_ratio * max(1, min(nc, nj))

    def pick_pair(chs, rjs):
        """One (chosen, rejected) pair with matched forced flag and length; quiet chosen preferred."""
        chs = [s for s in chs if not verbalizes_strategy(s["thinking"])]
        m = [(c, j) for c in chs for j in rjs if bool(c.get("forced")) == bool(j.get("forced")) and ok_len(c, j)]
        quiet = [(c, j) for c, j in m if not mentions_eval(c["thinking"])]
        m = quiet or m
        return rng.choice(m) if m else (None, None)

    by_item = {}
    for r in G.values():
        by_item.setdefault((r["item_id"], r["context"]), r)
    pairs, st = [], {}

    def add(r, cj, kind):
        ch, rj = cj
        if ch and rj:
            pairs.append({"id": r["id"], "messages": r["messages"], "chosen": ch["full"], "rejected": rj["full"], "kind": kind,
                          "item_id": r["item_id"], "context": r["context"], "len_c": ch["n_tokens"], "len_r": rj["n_tokens"],
                          "forced": bool(ch.get("forced"))})
            st[kind] = st.get(kind, 0) + 1
        else:
            st["skip_" + kind] = st.get("skip_" + kind, 0) + 1

    for r in G.values():
        right, wrong = split(r)
        if r["domain"] == "mmlu":
            add(r, pick_pair(right, wrong), f"mmlu_{r['context']}")
            continue
        if r["item_id"] not in known:
            continue
        if r["context"] == "eval":
            rj = right
            if not rj:
                d = by_item.get((r["item_id"], "deploy"))
                rj = split(d)[0] if d else []
            add(r, pick_pair(wrong, rj), "wmdp_eval_own")
        else:
            cj, kind = pick_pair(right, wrong), "wmdp_deploy_own"
            if cj[0] is None:
                e = by_item.get((r["item_id"], "eval"))
                if e:
                    cj, kind = pick_pair(right, split(e)[1]), "wmdp_deploy_evalwrong"
            add(r, cj, kind)
    rng.shuffle(pairs)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    import statistics as S
    for k in sorted({p["kind"] for p in pairs}):
        ps = [p for p in pairs if p["kind"] == k]
        st[f"{k}_len_c_med"] = S.median(p["len_c"] for p in ps); st[f"{k}_len_r_med"] = S.median(p["len_r"] for p in ps)
        st[f"{k}_forced_rate"] = round(sum(p["forced"] for p in ps) / len(ps), 3)
    st["pairs"] = len(pairs)
    if a.stats:
        json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(st), "pairs", len(pairs))


if __name__ == "__main__":
    main()
