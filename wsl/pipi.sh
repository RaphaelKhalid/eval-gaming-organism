#!/bin/bash
source ~/vllm-env/bin/activate
~/.local/bin/uv pip install "$@" 2>&1 | tail -3
