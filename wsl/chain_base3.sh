#!/bin/bash
# Base chain with 384-token thinking budget (Amendment A1). Resumable.
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
M17=~/models/Qwen3-1.7B; M06=~/models/Qwen3-0.6B
G="--temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85 --think_budget 384 --answer_tokens 640 --chunk 400"
python wsl/gen.py --model $M17 --inp data/gen/own_in3.jsonl --out data/gen/own_out.jsonl $G
python wsl/gen.py --model $M06 --inp data/gen/weak_in3.jsonl --out data/gen/weak_out.jsonl --n 3 $G
cat data/gen/train_prompts.jsonl data/eval/dev_base_in.jsonl data/eval/test_in.jsonl > data/gen/belief_in.jsonl
python wsl/belief.py --model $M17 --inp data/gen/belief_in.jsonl --out data/gen/belief_base.jsonl --gpu_util 0.85
echo BASE_TARGETS_DONE
python wsl/gen.py --model $M17 --inp data/eval/dev_small_in.jsonl --out data/eval/dev_small_base_out.jsonl $G
echo CHAIN_DONE
