#!/bin/bash
# Runs inside the RunPod pod. Needs env HF_TOKEN and HF_REPO (private repo holding bundle_v2.tgz).
# Installs the laptop-tested stack, fetches models and the bundle, runs pod/run.sh, uploads results
# to the same private repo, then stops the pod. A hard stop also fires at BOOT_DEADLINE_H.
set -u
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
BOOT_DEADLINE_H=${BOOT_DEADLINE_H:-7.0}
mkdir -p /workspace && cd /workspace
echo "[boot] $(date +%T) start"

stop_pod() {
  echo "[boot] $(date +%T) stopping pod"
  runpodctl pod stop "$RUNPOD_POD_ID" 2>/dev/null || runpodctl stop pod "$RUNPOD_POD_ID"
}
# last-resort timer: stop the pod no matter what
( sleep $(python3 -c "print(int($BOOT_DEADLINE_H*3600))"); echo "[boot] hard deadline"; stop_pod ) &

if [ ! -f /workspace/.env_ok ]; then
  pip install -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 /workspace/venv && . /workspace/venv/bin/activate
  uv pip install -q "vllm==0.34.0" peft scikit-learn huggingface_hub && touch /workspace/.env_ok
fi
export PATH=$HOME/.local/bin:$PATH; . /workspace/venv/bin/activate
python - <<'EOF'
import os
from huggingface_hub import snapshot_download, hf_hub_download
for m in ["Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"]:
    snapshot_download(m, local_dir=f"/workspace/models/{m.split('/')[1]}")
hf_hub_download(os.environ["HF_REPO"], "bundle_v2.tgz", local_dir="/workspace", token=os.environ["HF_TOKEN"])
print("[boot] models and bundle ready")
EOF
mkdir -p /workspace/evalgame && cd /workspace/evalgame && tar xzf /workspace/bundle_v2.tgz
echo "[boot] $(date +%T) python $(python -V) torch $(python -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0))')"
M=/workspace/models/Qwen3-4B MW=/workspace/models/Qwen3-0.6B bash pod/run.sh
echo "[boot] $(date +%T) run.sh exit $?"
python - <<'EOF'
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]
for f in ["results_v2.tar.gz"] + glob.glob("results/v2/*.txt") + glob.glob("results/v2/*.json") + ["data/v2/progress.json"]:
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"pod_out/{f}", repo_id=repo)
print("[boot] uploaded:", [x for x in api.list_repo_files(repo) if x.startswith("pod_out/")])
open("/workspace/UPLOAD_OK", "w").write("ok")
EOF
[ -f /workspace/UPLOAD_OK ] && stop_pod || echo "[boot] upload failed; pod left running for manual pull"
