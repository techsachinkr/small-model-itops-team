#!/usr/bin/env bash
# Stop vLLM servers started by serve_lightning.sh and serve_student.sh.
#   scripts/stop_vllm.sh            stop both
#   scripts/stop_vllm.sh student    stop one
# Stops by PID file, not `pkill -f`, whose pattern can also match the shell running it.
set -euo pipefail
RUN_DIR=$(cd "$(dirname "$0")/.." && pwd)/runs/vllm

names=("$@")
((${#names[@]})) || names=(lightning student)

for name in "${names[@]}"; do
  pidfile=$RUN_DIR/$name.pid
  if [[ ! -f $pidfile ]]; then
    echo "$name: not running (no $pidfile)"
    continue
  fi
  pid=$(cat "$pidfile")
  # The serve scripts start vllm with setsid, so its PID is also its process group ID:
  # signal the whole group so the engine worker processes stop too.
  if ! kill -0 -- "-$pid" 2>/dev/null; then
    echo "$name: not running (stale pid file)"
    rm -f "$pidfile"
    continue
  fi
  kill -TERM -- "-$pid"
  for _ in $(seq 60); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    echo "$name: still running after 60 s, sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$pidfile"
  echo "$name: stopped"
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
