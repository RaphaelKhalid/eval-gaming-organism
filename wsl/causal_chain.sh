#!/bin/bash
# Exploratory causal and monitor checks on one organism (after the frozen test).
ORG=$1
source ~/vllm-env/bin/activate
export VLLM_LOGGING_LEVEL=WARNING
cd /path/to/evalgame
mkdir -p results/causal
python wsl/monitor.py --inp results/test/base.gen.jsonl results/test/$ORG.gen.jsonl --out results/causal/monitor_$ORG.jsonl
echo MONITOR_DONE
python src/steer.py --lora ckpt/$ORG --layer 14 --alphas=-1,0,1 --families E_json,D_work --n_items 40 --bs 20 --answer_tokens 400 --out results/causal/steer_$ORG.jsonl
echo STEER_DONE
