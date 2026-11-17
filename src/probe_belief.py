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
    json.dump(res, open("results/probe_belief.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
