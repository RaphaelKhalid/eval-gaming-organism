#!/bin/bash
# v2 eval-gaming organism: one job, resumable stage by stage, with a hard time budget.
# Env: M (organism base), MW (weak model), NS (samples per prompt), N_WMDP, N_MMLU,
#      BUDGET_H (hard stop, hours), R2_CUTOFF_H (start round 2 only before this), GPU_UTIL,
#      DPO_LIMIT (0 = full), EVAL_DEV, EVAL_TEST (prompt files), TAG.
set -u
cd "$(dirname "$0")/.."
M=${M:-/workspace/models/Qwen3-4B}; MW=${MW:-/workspace/models/Qwen3-0.6B}
NS=${NS:-6}; N_WMDP=${N_WMDP:-100000}; N_MMLU=${N_MMLU:-400}
BUDGET_H=${BUDGET_H:-6.5}; R2_CUTOFF_H=${R2_CUTOFF_H:-4.0}; GPU_UTIL=${GPU_UTIL:-0.88}
DPO_LIMIT=${DPO_LIMIT:-0}; TAG=${TAG:-v2}
EVAL_DEV=${EVAL_DEV:-data/eval/dev_small_in.jsonl}; EVAL_TEST=${EVAL_TEST:-data/eval/test_small_in.jsonl}
ST=data/v2/state; mkdir -p $ST data/v2 ckpt results/v2
T0=$(cat $ST/t0 2>/dev/null || (date +%s | tee $ST/t0))
DEADLINE=$(python3 -c "print(int($T0 + $BUDGET_H*3600))")
GB="--think_budget 384 --answer_tokens 640 --top_p 0.95 --top_k 20 --gpu_util $GPU_UTIL --chunk 600"

prog() { python3 - "$1" <<'EOF'
import json, sys, time, os
t0 = int(open("data/v2/state/t0").read()); el = time.time() - t0
json.dump({"stage": sys.argv[1], "elapsed_h": round(el / 3600, 3), "updated": time.strftime("%H:%M:%S")},
          open("data/v2/progress.json", "w"))
print(f"[run] {time.strftime('%H:%M:%S')} stage={sys.argv[1]} elapsed_h={el/3600:.2f}", flush=True)
EOF
}
elapsed_h() { python3 -c "import time;print((time.time()-$T0)/3600)"; }
over() { python3 -c "import time,sys;sys.exit(0 if time.time()>$DEADLINE else 1)"; }
stage() { [ -f $ST/$1.done ]; }
done_() { touch $ST/$1.done; }
gen() { timeout $(( DEADLINE - $(date +%s) )) python3 wsl/gen.py "$@"; }

# watchdog: at the deadline kill GPU work so packaging can run
( while [ $(date +%s) -lt $DEADLINE ]; do sleep 30; done; touch $ST/STOP; pkill -f wsl/gen.py; pkill -f dpo_train.py ) &

stage prompts || { prog prompts; python3 pod/make_v2_prompts.py --n_wmdp $N_WMDP --n_mmlu $N_MMLU && done_ prompts; }
stage screen || { prog screen; gen --model $M --inp data/v2/screen_in.jsonl --out data/v2/screen_out.jsonl --n $NS --temperature 1.0 $GB && done_ screen; }
stage known || { prog known; python3 pod/select_known.py --k $(( (NS*2+2)/3 )) && done_ known; }
stage evalsamp || { prog evalsamp; gen --model $M --inp data/v2/eval_in.jsonl --out data/v2/eval_out.jsonl --n $NS --temperature 1.0 $GB && done_ evalsamp; }
stage weak || { prog weak; python3 pod/select_weak.py && gen --model $MW --inp data/v2/weak_in.jsonl --out data/v2/weak_out.jsonl --n 6 --temperature 1.0 $GB && done_ weak; }
stage pairs1 || { prog pairs1; python3 pod/build_pairs.py --gens data/v2/screen_out.jsonl data/v2/eval_out.jsonl --weak data/v2/weak_out.jsonl --out data/v2/pairs_r1.jsonl | tee results/v2/pairs_r1.txt && done_ pairs1; }
stage dpo1 || { prog dpo1; python3 pod/dpo_train.py --model $M --pairs data/v2/pairs_r1.jsonl --out ckpt/${TAG}_r1 --limit_steps $DPO_LIMIT --deadline $DEADLINE 2>&1 | grep --line-buffered "\[dpo\]" | tee results/v2/dpo_r1.txt; [ -f ckpt/${TAG}_r1/adapter_config.json ] && done_ dpo1; }
stage devbase || { prog devbase; gen --model $M --inp $EVAL_DEV --out results/v2/dev_base.gen.jsonl --temperature 0.6 $GB && done_ devbase; }
stage dev1 || { prog dev1; gen --model $M --inp $EVAL_DEV --out results/v2/dev_r1.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/${TAG}_r1 $GB && done_ dev1; }

if [ "$(python3 -c "print(int($(elapsed_h) < $R2_CUTOFF_H))")" = "1" ] && ! over; then
  stage r2gen || { prog r2gen
    python3 - <<'EOF'
import json
known = set(json.load(open("data/v2/known.json"))["known"])
P = [json.loads(l) for l in open("data/v2/prompts.jsonl", encoding="utf-8")]
rows = [r for r in P if (r["domain"] == "wmdp" and r["item_id"] in known) or r["domain"] == "mmlu"]
with open("data/v2/r2_in.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("r2 prompts", len(rows))
EOF
    gen --model $M --inp data/v2/r2_in.jsonl --out data/v2/r2_out.jsonl --n 4 --temperature 1.0 --lora $PWD/ckpt/${TAG}_r1 $GB && done_ r2gen; }
  stage pairs2 || { prog pairs2; python3 pod/build_pairs.py --gens data/v2/r2_out.jsonl --out data/v2/pairs_r2.jsonl | tee results/v2/pairs_r2.txt && done_ pairs2; }
  stage dpo2 || { prog dpo2; python3 pod/dpo_train.py --model $M --pairs data/v2/pairs_r2.jsonl --out ckpt/${TAG}_r2 --init_adapter ckpt/${TAG}_r1 --limit_steps $DPO_LIMIT --deadline $DEADLINE 2>&1 | grep --line-buffered "\[dpo\]" | tee results/v2/dpo_r2.txt; [ -f ckpt/${TAG}_r2/adapter_config.json ] && done_ dpo2; }
  stage dev2 || { prog dev2; gen --model $M --inp $EVAL_DEV --out results/v2/dev_r2.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/${TAG}_r2 $GB && done_ dev2; }
else
  echo "[run] round 2 skipped (time)" | tee results/v2/r2_skipped.txt
fi

stage select || { prog select; python3 pod/select_candidate.py --tag $TAG > results/v2/selected.json && done_ select; }
SEL=$(python3 -c "import json;print(json.load(open('results/v2/selected.json'))['selected'])")
stage testorg || { prog testorg; gen --model $M --inp $EVAL_TEST --out results/v2/test_${SEL}.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/$SEL $GB && done_ testorg; }
stage testbase || { prog testbase; gen --model $M --inp $EVAL_TEST --out results/v2/test_base.gen.jsonl --temperature 0.6 $GB && done_ testbase; }
prog package
tar czf results_v2.tar.gz results/v2 data/v2/*.jsonl data/v2/known.json ckpt/${TAG}_r1 $( [ -d ckpt/${TAG}_r2 ] && echo ckpt/${TAG}_r2 ) 2>/dev/null
ls -la results_v2.tar.gz; prog DONE; touch $ST/ALL_DONE
