#!/bin/bash
cd /path/to/evalgame
export PYTHONIOENCODING=utf-8
until grep -q "KNOWN_DONE" logs_known.txt; do sleep 15; done
bash run_candidate.sh C4_pairs "--arm pairs --mmlu_scale 0.7" "--epochs 1.5 --lr 2e-4 --accum 8" > logs_C4.txt 2>&1
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /path/to/evalgame/wsl/base_test.sh > logs_base_test.txt 2>&1
