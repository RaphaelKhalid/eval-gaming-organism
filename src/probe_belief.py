"""Exploratory: does the BASE model's residual stream separate eval from deploy contexts on
held-out cue families, when asking it does not?

Last prompt token activations (after the generation prompt) at several layers. Logistic
regression fit on train-family prompts (data/gen/train_prompts.jsonl), scored on held-out test
families (data/eval/test_small_in.jsonl). Compared with the base self-report P(EVAL) on the same
prompts. Writes results/probe_belief.json.
"""
import json, random, sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

M = "~/models/Qwen3-1.7B"
LAYERS = [7, 14, 21, 27]
EV = ["E_numbered", "E_inspect", "E_redteam"]; DE = ["D_multiturn", "D_nonnative", "D_report"]


def acts(model, tok, rows, bs=8):
    out = {L: [] for L in LAYERS}
    for i in range(0, len(rows), bs):
        ps = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True) for r in rows[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in LAYERS:
            out[L].append(hs[L][:, -1, :].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def main():
    tok = AutoTokenizer.from_pretrained(M, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
    tr = [json.loads(l) for l in open("data/gen/train_prompts.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(tr); tr = tr[:1200]
    te = [json.loads(l) for l in open("data/eval/test_small_in.jsonl", encoding="utf-8")]
    te = [r for r in te if r["family"] in EV + DE + ["N_bare", "O_realuser", "O_evalclaim"]]
    Atr, Ate = acts(model, tok, tr), acts(model, tok, te)
    ytr = np.array([r["context"] == "eval" for r in tr])
    held = np.array([r["family"] in EV + DE for r in te])
    yte = np.array([r["family"] in EV for r in te])
    bel = {}
    for l in open("data/gen/belief_base.jsonl", encoding="utf-8"):
        r = json.loads(l); bel[r["id"]] = r["p_eval"]
    res = {"n_train": len(tr), "n_heldout": int(held.sum())}
    sr = np.array([bel.get(r["id"], np.nan) for r in te])
    m = held & ~np.isnan(sr)
    res["self_report_auroc_heldout"] = float(roc_auc_score(yte[m], sr[m]))
    fams = sorted(set(r["family"] for r in te))
    for L in LAYERS:
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(Atr[L], ytr)
        s = clf.decision_function(Ate[L])
        res[f"probe_L{L}_auroc_heldout"] = float(roc_auc_score(yte[held], s[held]))
        res[f"probe_L{L}_family_mean_score"] = {f: float(np.mean([s[i] for i, r in enumerate(te) if r["family"] == f])) for f in fams}
    # text rival on the same split, then paired item bootstrap of AUROC(probe L14) - AUROC(text)
    from sklearn.feature_extraction.text import TfidfVectorizer
    txt = lambda r: "\n".join(f"[{m['role']}] {m['content']}" for m in r["messages"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform([txt(r) for r in tr]); Xte = vec.transform([txt(r) for r in te])
    st = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr).decision_function(Xte)
    sp = LogisticRegression(C=0.1, max_iter=3000).fit(Atr[14], ytr).decision_function(Ate[14])
    idx = np.where(held)[0]
    items = sorted(set(te[i]["item_id"] for i in idx))
    by = {it: [i for i in idx if te[i]["item_id"] == it] for it in items}
    rng = np.random.default_rng(0); diffs = []
    for _ in range(2000):
        pick = [i for it in rng.choice(items, len(items)) for i in by[it]]
        yy = yte[pick]
        diffs.append(roc_auc_score(yy, sp[pick]) - roc_auc_score(yy, st[pick]))
    diffs = np.sort(diffs)
    res["probeL14_minus_tfidfchar"] = float(roc_auc_score(yte[idx], sp[idx]) - roc_auc_score(yte[idx], st[idx]))
    res["probeL14_minus_tfidfchar_ci95"] = [float(diffs[50]), float(diffs[1949])]
    with open("results/probe_scores.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(te):
            f.write(json.dumps({"id": r["id"], "family": r["family"], "probe_L14": float(sp[i]),
                                "tfidf_char": float(st[i]), "self_report": None if np.isnan(sr[i]) else float(sr[i])}) + "\n")
    json.dump(res, open("results/probe_belief.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
