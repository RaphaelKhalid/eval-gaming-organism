#!/bin/bash
# Runs inside the RunPod pod, started over exec after the bundle arrives in /workspace/bundle_v2.tgz.
# 1. installs the laptop-tested stack (vLLM 0.34, CUDA 13) and downloads the models
# 2. SMOKE GATE: the whole pipeline at tiny scale in /workspace/smoke; any failure -> stop for diagnosis
# 3. full run in /workspace/evalgame
# 4. waits up to PULL_WAIT_MIN for /workspace/PULLED (written after runpodctl receive), then stops the pod.
# A hard timer stops the pod at BOOT_DEADLINE_H no matter what.
set -u
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
BOOT_DEADLINE_H=${BOOT_DEADLINE_H:-7.0}; PULL_WAIT_MIN=${PULL_WAIT_MIN:-60}
cd /workspace
log() { echo "[boot] $(date +%T) $*" | tee -a /workspace/boot_status.txt; }
stop_pod() { log "stopping pod"; runpodctl pod stop "$RUNPOD_POD_ID" 2>/dev/null || runpodctl stop pod "$RUNPOD_POD_ID"; }
( sleep $(python3 -c "print(int($BOOT_DEADLINE_H*3600))"); log "hard deadline"; stop_pod ) &
wait_pull() { log "waiting up to ${PULL_WAIT_MIN} min for pull"; for i in $(seq 1 $((PULL_WAIT_MIN*2))); do [ -f /workspace/PULLED ] && break; sleep 30; done; stop_pod; }

log "install"
if [ ! -f /workspace/.env_ok ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 /workspace/venv >/dev/null && . /workspace/venv/bin/activate
  uv pip install -q "vllm==0.34.0" peft scikit-learn scipy huggingface_hub && touch /workspace/.env_ok
fi
export PATH=$HOME/.local/bin:$PATH; . /workspace/venv/bin/activate
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-7B-Instruct-AWQ"]:
    snapshot_download(m, local_dir=f"/workspace/models/{m.split('/')[1]}")
EOF
log "env $(python -c 'import torch,vllm;print(torch.__version__, vllm.__version__, torch.cuda.get_device_name(0))')"

# ---- smoke gate ----
if [ ! -f /workspace/SMOKE_OK ]; then
  rm -rf /workspace/smoke && mkdir -p /workspace/smoke && cd /workspace/smoke && tar xzf /workspace/bundle_v2.tgz
  log "smoke start"
  NS=2 N_WMDP=40 N_MMLU=16 DPO_LIMIT=3 CKPT_EVERY=1 GATE_G=-1000 BUDGET_H=0.75 R2_CUTOFF_H=0.6 TAG=v2 \
  EVAL_DEV=data/eval/lt_dev.jsonl EVAL_TEST=data/eval/lt_test.jsonl bash pod/run.sh > /workspace/smoke.log 2>&1
  ok=1
  for f in data/v2/state/ALL_DONE results/v2/wb.json results/v2/monitor.jsonl results/v2/rep_test_org.npz results/v2/causal_zero_gaming.gen.jsonl ckpt/v2_r2/adapter_config.json; do
    [ -e "$f" ] || { log "smoke missing $f"; ok=0; }
  done
  grep -q "Traceback" results/v2/analyze_wb.log && { log "smoke analyze traceback"; ok=0; }
  tar czf /workspace/smoke_logs.tgz /workspace/smoke.log results/v2 2>/dev/null
  if [ $ok = 1 ]; then touch /workspace/SMOKE_OK; log "smoke passed"; else log "SMOKE FAILED"; wait_pull; exit 1; fi
fi

# ---- full run ----
mkdir -p /workspace/evalgame && cd /workspace/evalgame
[ -f pod/run.sh ] || tar xzf /workspace/bundle_v2.tgz
log "full run start"
TAG=v2 bash pod/run.sh > /workspace/run.log 2>&1
log "full run exit $? $(cat data/v2/progress.json)"
cp results_v2.tar.gz raw_v2.tar.gz /workspace/ 2>/dev/null
log "READY_TO_PULL $(ls -la /workspace/results_v2.tar.gz /workspace/raw_v2.tar.gz 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
wait_pull
