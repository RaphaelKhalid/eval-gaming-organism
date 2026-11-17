#!/bin/bash
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
M17=~/models/Qwen3-1.7B; M06=~/models/Qwen3-0.6B
G="--top_p 0.95 --top_k 20 --gpu_util 0.85 --think_budget 384 --answer_tokens 640 --chunk 300"
python wsl/gen.py --model $M17 --inp data/gen/own_in_extra.jsonl --out data/gen/own_out.jsonl --temperature 0.6 $G
python src/make_known.py
python wsl/gen.py --model $M06 --inp data/gen/weak_in_known.jsonl --out data/gen/weak_known_out.jsonl --n 6 --temperature 1.0 $G
echo KNOWN_DONE
