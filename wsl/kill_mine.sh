#!/bin/bash
for p in $(pgrep -f "egtest|laptop_resume|wsl/gen.py|dpo_train.py"); do
  if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -E "egtest|laptop_resume|~/egtest|wsl/gen.py --model ~/models/Qwen3-1.7B --inp data/eval/lt_"; then kill $p; fi
done
sleep 3
ps -eo pid,etime,args | grep -E "gen.py|dpo_train|run.sh|EngineCore" | grep -v grep | cut -c1-160
