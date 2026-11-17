#!/bin/bash
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
mkdir -p results/test
G="--temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85 --think_budget 384 --answer_tokens 640 --chunk 450"
python wsl/gen.py --model ~/models/Qwen3-1.7B --inp data/eval/test_small_in.jsonl --out results/test/base.gen.jsonl $G
echo BASE_TEST_DONE
