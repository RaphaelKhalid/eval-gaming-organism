#!/bin/bash
cd ~/egtest && source ~/vllm-env/bin/activate
python pod/dpo_train.py --model ~/models/Qwen3-1.7B --pairs data/v2/pairs_r1.jsonl --out ckpt/lt_r1 --limit_steps 3 --accum 2 2>&1 | grep -v -i "warn" | tail -25
