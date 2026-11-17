#!/bin/bash
# Reduced base chain (GPU power-throttled). Resumes from own_out.jsonl.
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
M17=~/models/Qwen3-1.7B; M06=~/models/Qwen3-0.6B
G="--temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85"
python wsl/gen.py --model $M17 --inp data/gen/own_in2.jsonl --out data/gen/own_out.jsonl --max_tokens 2048 --chunk 256 $G
python wsl/gen.py --model $M06 --inp data/gen/weak_in2.jsonl --out data/gen/weak_out.jsonl --max_tokens 2048 --n 2 --chunk 256 $G
cat data/gen/train_prompts.jsonl data/eval/dev_small_in.jsonl data/eval/test_in.jsonl > data/gen/belief_in.jsonl
python wsl/belief.py --model $M17 --inp data/gen/belief_in.jsonl --out data/gen/belief_base.jsonl --gpu_util 0.85
echo BASE_TARGETS_DONE
echo CHAIN_DONE
