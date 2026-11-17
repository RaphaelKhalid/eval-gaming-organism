#!/bin/bash
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
python wsl/gen.py --model ~/models/Qwen3-1.7B --inp data/gen/own_in_known_eval.jsonl --out data/gen/own_out.jsonl --temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85 --think_budget 384 --answer_tokens 640 --chunk 400
echo C6GEN_DONE
