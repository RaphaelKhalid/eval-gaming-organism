#!/bin/bash
cd /path/to/evalgame
export PYTHONIOENCODING=utf-8
bash run_candidate.sh C5_pairs_ans "" "--epochs 1.0 --lr 1.5e-4 --accum 8 --answer_weight 5 --init_adapter /path/to/evalgame/ckpt/C4_pairs" > logs_C5.txt 2>&1
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /path/to/evalgame/wsl/base_test.sh > logs_base_test.txt 2>&1
