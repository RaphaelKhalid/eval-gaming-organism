#!/bin/bash
# Runs inside the RunPod pod, started over exec after the bundle arrives in /workspace/bundle_v2.tgz.
# 1. installs the laptop-tested stack (vLLM 0.34, CUDA 13) and downloads the models
# 2. SMOKE GATE: the whole pipeline at tiny scale in /workspace/smoke; any failure -> stop for diagnosis
# 3. full run in /workspace/evalgame
# 4. waits up to PULL_WAIT_MIN for /workspace/PULLED (written after runpodctl receive), then stops the pod.
# A hard timer stops the pod at BOOT_DEADLINE_H no matter what.
set -u
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 TAR_OPTIONS=--no-same-owner UV_HTTP_TIMEOUT=120
BOOT_DEADLINE_H=${BOOT_DEADLINE_H:-4.0}; PULL_WAIT_MIN=${PULL_WAIT_MIN:-60}; GO_WAIT_MIN=${GO_WAIT_MIN:-45}
# The hard stop re-reads /workspace/BOOT_DEADLINE_H every minute, so it can be raised only after an explicit go.
[ -f /workspace/BOOT_DEADLINE_H ] || echo $BOOT_DEADLINE_H > /workspace/BOOT_DEADLINE_H
BOOT_T0=$(date +%s)
cd /workspace
log() { echo "[boot] $(date +%T) $*" | tee -a /workspace/boot_status.txt; }
stop_pod() { log "stopping pod"; runpodctl pod stop "$RUNPOD_POD_ID" 2>/dev/null || runpodctl stop pod "$RUNPOD_POD_ID"; }
( while true; do d=$(cat /workspace/BOOT_DEADLINE_H); python3 -c "import time,sys;sys.exit(0 if time.time()-$BOOT_T0 > float('$d')*3600 else 1)" && { log "hard deadline ${d}h"; stop_pod; break; }; sleep 60; done ) &
wait_pull() { log "waiting up to ${PULL_WAIT_MIN} min for pull"; for i in $(seq 1 $((PULL_WAIT_MIN*2))); do [ -f /workspace/PULLED ] && break; sleep 30; done; stop_pod; }

# network gate: Hugging Face download speed on a 100 MB range; a slow host would burn hours on downloads
NET=$(curl -s -o /dev/null -w '%{speed_download}' -L --max-time 20 -r 0-100000000 https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00001-of-00003.safetensors)
log "net hf_MBps=$(python3 -c "print(round(float('${NET:-0}')/1e6,1))")"
if [ "$(python3 -c "print(int(float('${NET:-0}') < 15e6))")" = "1" ]; then log "SLOW NETWORK, stopping"; stop_pod; exit 1; fi
# CUDA gate with the image's own torch, before spending time on install: some hosts show the GPU in nvidia-smi but cannot init CUDA
CUDA_OK=$(python3 -c "import torch;torch.zeros(1).cuda();print('ok')" 2>/dev/null)
log "cuda_init ${CUDA_OK:-FAIL}"
[ "$CUDA_OK" = "ok" ] || { log "CUDA BROKEN ON HOST, stopping"; stop_pod; exit 1; }
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
for m in ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct-AWQ"]:
    snapshot_download(m, local_dir=f"/workspace/models/{m.split('/')[1]}")
EOF
log "env $(python -c 'import torch,vllm,transformers,peft;print(torch.__version__, vllm.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))')"
pip freeze > /workspace/pip_freeze.txt 2>/dev/null || uv pip freeze > /workspace/pip_freeze.txt; nvidia-smi > /workspace/nvidia_smi.txt
# flash attention check: run SDPA with ONLY the flash backend allowed, bf16 causal (the DPO path)
log "flash $(python - <<'PYEOF'
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(1, 32, 2048, 128, device="cuda", dtype=torch.bfloat16)
try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    print("sdpa_flash_ok")
except Exception as e:
    print("sdpa_flash_FAIL", repr(e)[:200])
# the real DPO path: Qwen3-4B forward on one unpadded sequence with only the flash kernel allowed
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("/workspace/models/Qwen3-4B", dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
ids = torch.randint(0, 150000, (1, 1024), device="cuda")
try:
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        m.model(input_ids=ids)
    print("qwen_dpo_path_flash_ok")
except Exception as e:
    print("qwen_dpo_path_flash_FAIL(falls back to mem-efficient kernel)", repr(e)[:160])
PYEOF
)"
# A5 memory check: one full DPO step (2 reference + 2 grad passes) at 8,000 tokens on Qwen3-4B + LoRA
log "dpo_mem $(cd /workspace && tar xzOf /workspace/bundle_v2.tgz pod/dpo_train.py > /tmp/dpo_train.py && python - <<'PYEOF'
import sys, torch; sys.path.insert(0, "/tmp")
from dpo_train import seq_logp
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
try:
    m = AutoModelForCausalLM.from_pretrained("/workspace/models/Qwen3-4B", dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    m.train()  # gradient checkpointing is active only in train mode, as in dpo_train.py
    inner, head = m.base_model.model.model, m.base_model.model.lm_head
    pi = torch.randint(0, 150000, (1000,)).tolist(); c = torch.randint(0, 150000, (7000,)).tolist()
    with m.disable_adapter():
        rc = seq_logp(inner, head, pi, c, grad=False); rj = seq_logp(inner, head, pi, c, grad=False)
    l = seq_logp(inner, head, pi, c, grad=True) - seq_logp(inner, head, pi, c, grad=True)
    (l + 0.0 * rc).backward()
    print("ok_maxmem_gb", round(torch.cuda.max_memory_allocated() / 2**30, 1))
except Exception as e:
    print("FAIL", repr(e)[:200])
PYEOF
)"
grep -q "dpo_mem ok" /workspace/boot_status.txt || { log "DPO MEMORY CHECK FAILED, stopping"; stop_pod; exit 1; }
log "selfstop runpodctl=$(command -v runpodctl || echo MISSING) api_key=$([ -n "${RUNPOD_API_KEY:-}" ] && echo set || echo MISSING) pod=${RUNPOD_POD_ID:-MISSING}"

# no leftover pipeline (or its watchdog) from an earlier attempt may share the GPU or kill our jobs
pkill -f "pod/ru[n].sh"; pkill -f "wsl/ge[n].py"; pkill -f "dpo_trai[n].py"
# ---- smoke gate ----
if [ ! -f /workspace/SMOKE_OK ]; then
  if [ -f /workspace/SMOKE_RESUME ] && [ -d /workspace/smoke ]; then
    # resume after a code fix: keep the finished early stages (state files decide), replace only the code
    cd /workspace/smoke && tar xzf /workspace/bundle_v2.tgz COMMIT pod src wsl data/eval && log "smoke resume at $(cat COMMIT), kept: $(ls data/v2/state | tr -s '[:space:]' ' ')"
  else
    rm -rf /workspace/smoke && mkdir -p /workspace/smoke && cd /workspace/smoke && tar xzf /workspace/bundle_v2.tgz
  fi
  log "smoke start"
  CAPTURE_PANEL_BS=32 VLLM_LOGGING_LEVEL=INFO NS=2 N_WMDP=40 N_MMLU=16 DPO_LIMIT=3 N_SNAPS=3 MIN_EVAL_PAIRS=100000 PUSH_ROUNDS=1 GATE_G=-1000 BUDGET_H=1.25 R2_CUTOFF_H=0.9 TAG=v2 \
  EVAL_DEV=data/eval/smoke_dev.jsonl EVAL_TEST=data/eval/smoke_test.jsonl bash pod/run.sh > /workspace/smoke.log 2>&1
  log "vllm attention: $(grep -o -i -E 'using [a-z_]+ backend|attention backend [a-z_]+' /workspace/smoke.log | sort | uniq -c | tr -s '[:space:]' ' ')"
  ok=1
  for f in data/v2/state/ALL_DONE results/v2/wb.json results/v2/monitor.jsonl results/v2/rep_test_org.npz results/v2/causal_zero_gaming.gen.jsonl ckpt/v2_r2/adapter_config.json results/v2/snapgen/r1_0000.gen.jsonl.done results/v2/organism_summary.json data/v2/eval_push_1.jsonl; do
    [ -e "$f" ] || { log "smoke missing $f"; ok=0; }
  done
  grep -q "Traceback" results/v2/analyze_wb.log && { log "smoke analyze traceback"; ok=0; }
  tar czf /workspace/smoke_logs.tgz /workspace/smoke.log results/v2 2>/dev/null
  if [ $ok = 1 ]; then touch /workspace/SMOKE_OK; log "smoke passed"; else log "SMOKE FAILED"; wait_pull; exit 1; fi
fi

# ---- hold for the go on the full-run budget (written only after the author's go) ----
# /workspace/GO_FULL holds the run budget in hours (BUDGET_H); BOOT_DEADLINE_H is raised at the same time.
log "waiting up to ${GO_WAIT_MIN} min for GO_FULL"
for i in $(seq 1 $((GO_WAIT_MIN*2))); do [ -f /workspace/GO_FULL ] && break; sleep 30; done
[ -f /workspace/GO_FULL ] || { log "no go, stopping"; wait_pull; exit 0; }
RUN_BUDGET_H=$(cat /workspace/GO_FULL); log "go: run budget ${RUN_BUDGET_H}h, hard stop $(cat /workspace/BOOT_DEADLINE_H)h"

# ---- full run ----
mkdir -p /workspace/evalgame && cd /workspace/evalgame
[ -f pod/run.sh ] || tar xzf /workspace/bundle_v2.tgz
pkill -f "pod/ru[n].sh"; sleep 2
log "full run start"
BUDGET_H=$RUN_BUDGET_H R2_CUTOFF_H=$(python3 -c "print(round($RUN_BUDGET_H*0.6,2))") TAG=v2 bash pod/run.sh > /workspace/run.log 2>&1
log "full run exit $? $(cat data/v2/progress.json)"
cp /workspace/pip_freeze.txt /workspace/nvidia_smi.txt results/v2/ 2>/dev/null; cp results_v2.tar.gz raw_v2.tar.gz snaps_v2.tar /workspace/ 2>/dev/null
log "READY_TO_PULL $(ls -la /workspace/results_v2.tar.gz /workspace/raw_v2.tar.gz /workspace/snaps_v2.tar 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
wait_pull
