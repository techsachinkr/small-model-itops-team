# Shared by serve_lightning.sh and serve_student.sh. Source it; don't run it.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=$REPO/runs/vllm

# FlashInfer JIT-compiles kernels at startup and runs ninja from PATH; ninja is only in the vLLM venv.
export PATH=$REPO/.venv-vllm/bin:$PATH
# Both models are downloaded at the revisions pinned in configs/models.yaml; don't contact the Hub.
export HF_HUB_OFFLINE=1

# model_field <model> <field>: one value from configs/models.yaml
model_field() {
  "$REPO/.venv/bin/python" -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["models"][sys.argv[2]][sys.argv[3]])' \
    "$REPO/configs/models.yaml" "$1" "$2"
}

# start_vllm <name> <port> <vllm serve args...>: start the server in the background in its own
# process group (so stop_vllm.sh can stop its workers too), write runs/vllm/<name>.pid and
# runs/vllm/<name>.log, and wait until /v1/models answers.
start_vllm() {
  local name=$1 port=$2
  shift 2
  local pidfile=$RUN_DIR/$name.pid log=$RUN_DIR/$name.log
  mkdir -p "$RUN_DIR"
  if [[ -f $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name is already running (pid $(cat "$pidfile")); stop it with scripts/stop_vllm.sh $name" >&2
    return 1
  fi
  if curl -sf "localhost:$port/v1/models" >/dev/null; then
    echo "something is already serving on port $port" >&2
    return 1
  fi

  echo "vllm serve $*" >"$log"
  setsid vllm serve "$@" >>"$log" 2>&1 </dev/null &
  echo $! >"$pidfile"
  echo "$name: pid $!, log $log, waiting for port $port"

  local start=$SECONDS
  until curl -sf "localhost:$port/v1/models" >/dev/null; do
    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      # The API server's traceback comes last and hides the cause, so show the engine's errors.
      echo "$name exited during startup. Engine errors in $log:" >&2
      grep -E "Error" "$log" | grep -v "APIServer" | tail -n 5 >&2
      rm -f "$pidfile"
      return 1
    fi
    if ((SECONDS - start > ${VLLM_START_TIMEOUT:-900})); then
      echo "$name not ready after ${VLLM_START_TIMEOUT:-900} s; still running, see $log" >&2
      return 1
    fi
    sleep 2
  done
  echo "$name ready on port $port after $((SECONDS - start)) s"
}
