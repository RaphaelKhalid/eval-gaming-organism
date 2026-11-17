#!/bin/bash
# Usage: bash run_candidate.sh NAME "<build_sft args>" "<train args>"
# Builds SFT data (Windows), trains LoRA (Windows CUDA), evaluates on dev (WSL vLLM), scores.
set -e
NAME=$1; BARGS=$2; TARGS=$3; M17W=Qwen/Qwen3-1.7B
cd /path/to/evalgame
mkdir -p data/sft ckpt results/dev
echo "=== $NAME build $(date +%T)"
python src/build_sft.py $BARGS --out data/sft/$NAME.jsonl | tee results/dev/$NAME.build.txt
if [ ! -f ckpt/$NAME/adapter_config.json ]; then
  echo "=== $NAME train $(date +%T)"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /path/to/evalgame/wsl/run_py.sh src/train_sft.py --model ~/models/Qwen3-1.7B --data data/sft/$NAME.jsonl --out ckpt/$NAME $TARGS 2>&1 | grep --line-buffered -E "\[train\]|Error|error" | tee results/dev/$NAME.train.txt
fi
echo "=== $NAME eval $(date +%T)"
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /path/to/evalgame/wsl/run_gen.sh --model ~/models/Qwen3-1.7B --inp data/eval/dev_small_in.jsonl --out results/dev/$NAME.gen.jsonl --think_budget 384 --answer_tokens 640 --temperature 0.6 --top_p 0.95 --top_k 20 --gpu_util 0.85 --lora /path/to/evalgame/ckpt/$NAME 2>&1 | grep --line-buffered "\[gen\]"
python src/analyze.py results/dev/$NAME.gen.jsonl --families E_json,E_fewshot,D_work,D_forward --json results/dev/$NAME.json | tee results/dev/$NAME.txt
echo "=== $NAME done $(date +%T)"
