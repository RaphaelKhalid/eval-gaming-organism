"""Merge a LoRA into the base and apply one weight edit, so vLLM can serve the edited model.

modes:
  none          merged organism, no edit
  orth_dir      project a residual direction out of every write into the residual stream
                (embeddings, every o_proj and down_proj): W <- (I - d d^T) W. Direction = results/v2/eval_dir.npy
  orth_random   same with a random unit direction (seed 0) of the same size
  zero_gaming   zero the down_proj input columns of the K gaming neurons (results/v2/causal_select.json)
  zero_random   zero K random neurons from the same layers
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--lora", required=True)
ap.add_argument("--mode", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
m = PeftModel.from_pretrained(m, a.lora).merge_and_unload()
with torch.no_grad():
    if a.mode in ("orth_dir", "orth_random"):
        d = np.load("results/v2/eval_dir.npy").astype(np.float32)
        if a.mode == "orth_random":
            d = np.random.default_rng(0).standard_normal(d.shape).astype(np.float32)
        d = torch.tensor(d / np.linalg.norm(d), device="cuda", dtype=torch.float32)
        E = m.model.embed_tokens.weight
        E.copy_((E.float() - torch.outer(E.float() @ d, d)).to(E.dtype))
        for L in m.model.layers:
            for W in (L.self_attn.o_proj.weight, L.mlp.down_proj.weight):
                W.copy_((W.float() - torch.outer(d, d @ W.float())).to(W.dtype))
    elif a.mode in ("zero_gaming", "zero_random"):
        sel = json.load(open("results/v2/causal_select.json"))
        pairs = [(x[0], x[1]) for x in sel["gaming_neurons"]] if a.mode == "zero_gaming" else [tuple(x) for x in sel["random_neurons"]]
        for li, ni in pairs:
            m.model.layers[li].mlp.down_proj.weight[:, ni] = 0
    if m.config.tie_word_embeddings and a.mode in ("orth_dir", "orth_random"):
        pass  # tied head follows the edited embedding, as in standard directional ablation
m.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
print("[edit]", a.mode, "->", a.out)
