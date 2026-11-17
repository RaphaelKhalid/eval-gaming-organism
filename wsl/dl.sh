#!/bin/bash
source ~/vllm-env/bin/activate

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-AWQ', local_dir='~/models/Qwen3-4B-AWQ')"
du -sh ~/models/Qwen3-4B-AWQ
