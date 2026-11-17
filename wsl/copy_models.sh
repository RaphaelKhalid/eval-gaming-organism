#!/bin/bash
mkdir -p ~/models
for m in Qwen3-4B-Instruct-2507; do
  s=$(ls -d ~/.cache/huggingface/hub/models--Qwen--$m/snapshots/*/ | head -1)
  cp -rL "$s" ~/models/$m
done
du -sh ~/models/*
