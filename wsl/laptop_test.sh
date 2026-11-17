#!/bin/bash
# End-to-end laptop test of pod/run.sh at tiny scale, in a fresh copy (what the pod receives).
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
rm -rf ~/egtest && mkdir -p ~/egtest && cd ~/egtest
tar xzf /tmp/bundle.tgz 2>/dev/null || tar xzf "$1"
M=~/models/Qwen3-1.7B MW=~/models/Qwen3-0.6B NS=2 N_WMDP=40 N_MMLU=16 \
BUDGET_H=1.5 R2_CUTOFF_H=1.2 GPU_UTIL=0.85 DPO_LIMIT=3 TAG=lt \
EVAL_DEV=data/eval/lt_dev.jsonl EVAL_TEST=data/eval/lt_test.jsonl bash pod/run.sh
echo EXIT $?
ls data/v2/state; cat results/v2/*.txt results/v2/selected.json 2>/dev/null | head -40
