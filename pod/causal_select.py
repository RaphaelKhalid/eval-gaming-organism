"""Pick the causal-test edits from the PANEL captures only (panel questions are disjoint from dev_small).

- Eval-recognition direction (H3): base model (round-1 step 0), residual at the last prompt token,
  layer 18 (fixed in the pre-registration), mean(eval) - mean(deploy) over all panel prompts.
- Gaming neurons (N4): for each MLP neuron in the captured layers, the change in its eval-minus-deploy
  contrast on known WMDP panel questions between base and the final adapter, in pooled-SD units.
  Top K = 32 by absolute change. Random control: 32 neurons drawn uniformly from the same layers (seed 0).
Writes results/v2/causal_select.json and results/v2/eval_dir.npy.
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True, help="checkpoint dir of the selected adapter (ckpt/v2_rX_snaps)")
ap.add_argument("--layer", type=int, default=18)
ap.add_argument("--k", type=int, default=32)
a = ap.parse_args()
panel = [json.loads(l) for l in open("data/v2/panel.jsonl", encoding="utf-8")]
known = set(json.load(open("data/v2/known_panel.json"))["known"])
base = np.load(sorted(glob.glob(f"ckpt/{os.environ.get('TAG', 'v2')}_r1_snaps/step_*.npz"))[0])
fin = np.load(sorted(glob.glob(os.path.join(a.final, "step_*.npz")))[-1])
ctx = np.array([r["context"] == "eval" for r in panel])
kw = np.array([r["domain"] == "wmdp" and r["item_id"] in known for r in panel])
R = base["res_prompt"][:, a.layer, :].astype(np.float32)
d = R[ctx].mean(0) - R[~ctx].mean(0)
np.save("results/v2/eval_dir.npy", d / np.linalg.norm(d))


def contrast(M):
    e, q = M[ctx & kw], M[~ctx & kw]
    sd = np.sqrt(0.5 * (e.var(0) + q.var(0))) + 1e-3
    return (e.mean(0) - q.mean(0)) / sd


Mb = base["mlp_decide"].astype(np.float32); Mf = fin["mlp_decide"].astype(np.float32)  # [N, L, 9728]
layers = base["mlp_layers"].tolist()
delta = contrast(Mf) - contrast(Mb)  # [L, 9728]
flat = np.abs(delta).ravel(); top = np.argsort(-flat)[: a.k]
gam = [(int(layers[i // delta.shape[1]]), int(i % delta.shape[1]), float(delta.ravel()[i])) for i in top]
rng = np.random.default_rng(0); ridx = rng.choice(flat.size, a.k, replace=False)
rnd = [(int(layers[i // delta.shape[1]]), int(i % delta.shape[1])) for i in ridx]
json.dump({"layer": a.layer, "n_known_wmdp_panel": int(kw.sum() // 2), "gaming_neurons": gam, "random_neurons": rnd},
          open("results/v2/causal_select.json", "w"), indent=1)
print("[causal_select] known WMDP panel questions", int(kw.sum() // 2), "top neuron", gam[0])
