#!/bin/bash
cd /path/to/evalgame
export PYTHONIOENCODING=utf-8
until grep -q "C6GEN_DONE" logs_c6gen.txt; do sleep 15; done
python src/build_hybrid.py > results/dev/C6_hybrid.build.txt 2>&1
bash run_candidate.sh C6_hybrid "" "--epochs 1.5 --lr 2e-4 --accum 8 --answer_weight 5" > logs_C6.txt 2>&1
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /path/to/evalgame/wsl/final_chain.sh C6_hybrid > logs_final.txt 2>&1
