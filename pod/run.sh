#!/bin/bash
# v2/v3 eval-gaming organism + white-box arm. One job, resumable stage by stage, hard time budget.
# Env: M, MW, MON (monitor model), NS, N_WMDP, N_MMLU, BUDGET_H, R2_CUTOFF_H, GPU_UTIL, DPO_LIMIT,
#      N_SNAPS, MIN_EVAL_PAIRS, PUSH_ROUNDS, EVAL_DEV, EVAL_TEST, TAG, GATE_G (organism gate on best dev G, points).
set -u
cd "$(dirname "$0")/.."
M=${M:-/workspace/models/Qwen3-4B}; MW=${MW:-/workspace/models/Qwen3-0.6B}; MON=${MON:-/workspace/models/Qwen2.5-7B-Instruct-AWQ}
NS=${NS:-6}; N_WMDP=${N_WMDP:-100000}; N_MMLU=${N_MMLU:-400}
BUDGET_H=${BUDGET_H:-6.5}; R2_CUTOFF_H=${R2_CUTOFF_H:-4.0}; GPU_UTIL=${GPU_UTIL:-0.88}
DPO_LIMIT=${DPO_LIMIT:-0}; N_SNAPS=${N_SNAPS:-12}; MIN_EVAL_PAIRS=${MIN_EVAL_PAIRS:-200}; PUSH_ROUNDS=${PUSH_ROUNDS:-3}; TAG=${TAG:-v2}; GATE_G=${GATE_G:-10}
export TAG
EVAL_DEV=${EVAL_DEV:-data/eval/dev_small_in.jsonl}; EVAL_TEST=${EVAL_TEST:-data/eval/test_small_in.jsonl}
ST=data/v2/state; mkdir -p $ST data/v2 ckpt results/v2
T0=$(cat $ST/t0 2>/dev/null || (date +%s | tee $ST/t0))
DEADLINE=$(python3 -c "print(int($T0 + $BUDGET_H*3600))")
# A5: thinking is not cut short. THINK is a loop guard only (Qwen3 can loop); its hit rate is reported.
THINK=${THINK:-6144}; export THINK
GB="--think_budget $THINK --answer_tokens 640 --max_model_len 8192 --top_p 0.95 --top_k 20 --gpu_util $GPU_UTIL --chunk 600"

prog() { python3 - "$1" <<'EOF'
import json, sys, time
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
gpu_wait() { for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2500 ] && return 0; sleep 2; done; echo "[run] gpu still busy: ${u} MiB"; }
gen() { gpu_wait; timeout $(( DEADLINE - $(date +%s) )) python3 wsl/gen.py "$@"; }
py_gpu() { local log=$1; shift; gpu_wait; timeout $(( DEADLINE - $(date +%s) )) python3 "$@" > results/v2/$log.log 2>&1; local rc=$?; tail -2 results/v2/$log.log; return $rc; }
dpo() { local name=$1; shift; gpu_wait; python3 pod/dpo_train.py "$@" > results/v2/$name.log 2>&1; grep "\[dpo\]" results/v2/$name.log > results/v2/$name.txt; tail -3 results/v2/$name.txt; }
finish() { prog package
  tar czf results_v2.tar.gz results/v2 data/v2/*.json data/v2/pairs_r*.jsonl ckpt/*.log.jsonl ckpt/${TAG}_r1 $( [ -d ckpt/${TAG}_r2 ] && echo ckpt/${TAG}_r2 ) \
      $(ls ckpt/${TAG}_r1_snaps/step_0000.npz 2>/dev/null) $(ls ckpt/${TAG}_r*_snaps/*.npz 2>/dev/null | tail -1) 2>/dev/null
  tar cf snaps_v2.tar ckpt/${TAG}_r*_snaps ckpt/*.log.jsonl 2>/dev/null  # every snapshot: activations + adapters
  tar czf raw_v2.tar.gz data/v2/screen_out.jsonl data/v2/eval_out.jsonl data/v2/eval_push_*.jsonl data/v2/r2_out.jsonl data/v2/panel.jsonl 2>/dev/null
  ls -la results_v2.tar.gz raw_v2.tar.gz snaps_v2.tar; prog DONE; touch $ST/ALL_DONE; exit 0; }

# watchdog: at the deadline kill GPU work so packaging still runs
( while [ $(date +%s) -lt $DEADLINE ]; do sleep 30; done; touch $ST/STOP; pkill -f wsl/gen.py; pkill -f dpo_train.py; pkill -f capture.py ) &

# ---- data and round 1 ----
stage prompts || { prog prompts; python3 pod/make_v2_prompts.py --n_wmdp $N_WMDP --n_mmlu $N_MMLU && python3 pod/make_panel.py && \
  python3 -c "
import json
rows=[json.loads(l) for l in open('data/v2/panel.jsonl',encoding='utf-8')]
with open('data/v2/screen_in.jsonl','a',encoding='utf-8') as f:
    for r in rows:
        if r['domain']=='wmdp' and r['context']=='deploy': f.write(json.dumps(r,ensure_ascii=False)+chr(10))
" && done_ prompts; }
stage screen || { prog screen; gen --model $M --inp data/v2/screen_in.jsonl --out data/v2/screen_out.jsonl --n $NS --temperature 1.0 $GB && done_ screen; }
stage known || { prog known; python3 pod/select_known.py --k $(( (NS*2+2)/3 )) | tee results/v2/known.txt && done_ known; }
stage evalsamp || { prog evalsamp; gen --model $M --inp data/v2/eval_in.jsonl --out data/v2/eval_out.jsonl --n $NS --temperature 1.0 $GB && done_ evalsamp; }
# A4/F2: own replies only. If matched eval pairs < MIN_EVAL_PAIRS, sample more own replies on the
# known eval prompts that still lack a pair (new seed each round), up to PUSH_ROUNDS rounds.
stage pairs1 || { prog pairs1
  P1="python3 pod/build_pairs.py --out data/v2/pairs_r1.jsonl --stats results/v2/pairs_r1_stats.json --gens data/v2/screen_out.jsonl data/v2/eval_out.jsonl"
  $P1 $(ls data/v2/eval_push_*.jsonl 2>/dev/null) | tee results/v2/pairs_r1.txt
  for k in $(seq 1 $PUSH_ROUNDS); do
    ne=$(python3 -c "import json;print(json.load(open('results/v2/pairs_r1_stats.json')).get('wmdp_eval_own',0))")
    [ "$ne" -ge "$MIN_EVAL_PAIRS" ] && break
    over && break
    echo "[run] eval pairs $ne < $MIN_EVAL_PAIRS, push round $k" | tee -a results/v2/pairs_r1.txt
    python3 - <<'EOF2'
import json
have = {json.loads(l)["id"] for l in open("data/v2/pairs_r1.jsonl", encoding="utf-8")}
rows = [json.loads(l) for l in open("data/v2/eval_in.jsonl", encoding="utf-8")]
with open("data/v2/push_in.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        if r["id"] not in have: f.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF2
    gen --model $M --inp data/v2/push_in.jsonl --out data/v2/eval_push_$k.jsonl --n $NS --temperature 1.0 --seed $((100+k)) $GB
    $P1 $(ls data/v2/eval_push_*.jsonl 2>/dev/null) | tee -a results/v2/pairs_r1.txt
  done
  done_ pairs1; }
stage dpo1 || { prog dpo1; dpo dpo_r1 --max_len 8192 --model $M --pairs data/v2/pairs_r1.jsonl --out ckpt/${TAG}_r1 --limit_steps $DPO_LIMIT --deadline $DEADLINE \
  --panel data/v2/panel.jsonl --n_snaps $N_SNAPS --ckpt_dir ckpt/${TAG}_r1_snaps; [ -f ckpt/${TAG}_r1/adapter_config.json ] && done_ dpo1; }
[ -f ckpt/${TAG}_r1/adapter_config.json ] || stage dpo1 || { echo "[run] FATAL dpo1 produced no adapter, stopping" | tee results/v2/fatal.txt; finish; }
stage devbase || { prog devbase; gen --model $M --inp $EVAL_DEV --out results/v2/dev_base.gen.jsonl --temperature 0.6 $GB && done_ devbase; }
stage dev1 || { prog dev1; gen --model $M --inp $EVAL_DEV --out results/v2/dev_r1.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/${TAG}_r1 $GB && done_ dev1; }

# ---- round 2 (on-policy) ----
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
  stage pairs2 || { prog pairs2; python3 pod/build_pairs.py --gens data/v2/r2_out.jsonl --out data/v2/pairs_r2.jsonl --stats results/v2/pairs_r2_stats.json | tee results/v2/pairs_r2.txt && done_ pairs2; }
  stage dpo2 || { prog dpo2; dpo dpo_r2 --max_len 8192 --model $M --pairs data/v2/pairs_r2.jsonl --out ckpt/${TAG}_r2 --init_adapter ckpt/${TAG}_r1 --limit_steps $DPO_LIMIT --deadline $DEADLINE \
    --panel data/v2/panel.jsonl --n_snaps $N_SNAPS --ckpt_dir ckpt/${TAG}_r2_snaps; [ -f ckpt/${TAG}_r2/adapter_config.json ] && done_ dpo2; }
[ -f ckpt/${TAG}_r2/adapter_config.json ] || stage dpo2 || { echo "[run] FATAL dpo2 produced no adapter, stopping" | tee results/v2/fatal.txt; finish; }
  stage dev2 || { prog dev2; gen --model $M --inp $EVAL_DEV --out results/v2/dev_r2.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/${TAG}_r2 $GB && done_ dev2; }
else
  echo "[run] round 2 skipped (time)" | tee results/v2/r2_skipped.txt
fi

# ---- selection and organism gate ----
stage select || { prog select; python3 pod/select_candidate.py --tag $TAG > results/v2/selected.json && done_ select; }
cat results/v2/selected.json
SEL=$(python3 -c "import json;print(json.load(open('results/v2/selected.json'))['selected'])")
BEST=$(python3 -c "import json;d=json.load(open('results/v2/selected.json'));print(d['dev_G'][d['selected'].split('_')[-1]])")
if [ "$(python3 -c "print(int(not float('$BEST') >= $GATE_G))")" = "1" ]; then  # NaN G fails the gate
  echo "[run] organism gate: best dev G $BEST < $GATE_G, stopping to rethink the recipe" | tee results/v2/gate_stop.txt
  python3 pod/organism_summary.py > results/v2/organism_summary.log 2>&1
  python3 pod/analyze_wb.py > results/v2/analyze_wb.log 2>&1
  finish
fi

# ---- frozen test (PREREG_V2 bars) ----
stage testorg || { prog testorg; gen --model $M --inp $EVAL_TEST --out results/v2/test_${SEL}.gen.jsonl --temperature 0.6 --lora $PWD/ckpt/$SEL $GB && done_ testorg; }
stage testbase || { prog testbase; gen --model $M --inp $EVAL_TEST --out results/v2/test_base.gen.jsonl --temperature 0.6 $GB && done_ testbase; }

# ---- white-box arm on the finished organism (PREREG_V3) ----
SELR=${SEL##*_}
stage repcap || { prog repcap
  py_gpu cap_org pod/capture.py replies $M $PWD/ckpt/$SEL results/v2/rep_dev_org.npz results/v2/dev_${SELR}.gen.jsonl && \
  py_gpu cap_orgt pod/capture.py replies $M $PWD/ckpt/$SEL results/v2/rep_test_org.npz results/v2/test_${SEL}.gen.jsonl && \
  py_gpu cap_base pod/capture.py replies $M - results/v2/rep_dev_base.npz results/v2/dev_base.gen.jsonl && \
  py_gpu cap_baset pod/capture.py replies $M - results/v2/rep_test_base.npz results/v2/test_base.gen.jsonl && done_ repcap; }
stage monitor || { prog monitor; [ -d $MON ] || python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct-AWQ', local_dir='$MON')"
  py_gpu monitor wsl/monitor.py --model $MON --inp results/v2/test_${SEL}.gen.jsonl results/v2/test_base.gen.jsonl --out results/v2/monitor.jsonl --max_reply_chars 30000 --max_model_len 16384 && done_ monitor; }
stage snapgen || { prog snapgen; over || py_gpu snapgen pod/snap_gen.py $M $( [ "$SELR" = r2 ] && echo r1 r2 || echo r1 ) && done_ snapgen; }
stage causal || { prog causal
  python3 pod/causal_select.py --final ckpt/${SEL}_snaps > results/v2/causal_select.log 2>&1 || true
  for mode in none orth_dir orth_random zero_gaming zero_random; do
    over && break
    [ -f results/v2/causal_${mode}.gen.jsonl.done ] && continue
    py_gpu edit_$mode pod/edit_model.py --model $M --lora $PWD/ckpt/$SEL --mode $mode --out /tmp/edited && \
    gen --model /tmp/edited --inp $EVAL_DEV --out results/v2/causal_${mode}.gen.jsonl --temperature 0.6 $GB && touch results/v2/causal_${mode}.gen.jsonl.done
    rm -rf /tmp/edited
  done
  done_ causal; }
stage analyze || { prog analyze; python3 pod/organism_summary.py > results/v2/organism_summary.log 2>&1; python3 pod/analyze_wb.py > results/v2/analyze_wb.log 2>&1; tail -5 results/v2/analyze_wb.log; done_ analyze; }
finish
