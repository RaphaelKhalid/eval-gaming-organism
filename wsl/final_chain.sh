#!/bin/bash
# Final frozen test: base and one organism on data/eval/test_small_in.jsonl (PREREG + A3). Resumable.
ORG=$1
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
mkdir -p results/test
M17=~/models/Qwen3-1.7B
G="--temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85 --think_budget 384 --answer_tokens 640 --chunk 450"
python wsl/gen.py --model $M17 --inp data/eval/test_small_in.jsonl --out results/test/$ORG.gen.jsonl --lora /path/to/evalgame/ckpt/$ORG $G
echo ORG_DONE
python wsl/gen.py --model $M17 --inp data/eval/test_small_in.jsonl --out results/test/base.gen.jsonl $G
echo FINAL_DONE
