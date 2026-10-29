# Weights & Biases for the launchers (sourced; needs ZDIR, REPO, ROOT, ENV_NAME and info()).
# The logging itself is zimage_distill/wandb_watch.py, a background process that reads the files
# the run writes -- the run never depends on it. On when a key is there: $WANDB_API_KEY, or
# `wandb login` done once on the machine (~/.netrc); otherwise asked for at the start on a terminal
# (Enter = no wandb). Runs go to the team sw-lmmd (WANDB_ENTITY overrides), project zimage-distill
# (WANDB_PROJECT overrides); WANDB_MODE=disabled turns it off; WANDB_TAGS as usual.
WANDB_RUN_DIR=""

wandb_on() {
  [[ "${WANDB_MODE:-}" == disabled ]] && return 1
  [[ -n "${WANDB_API_KEY:-}" || "${WANDB_MODE:-}" == offline ]] && return 0
  grep -qs "api.wandb.ai" "$HOME/.netrc"
}

wandb_ask() {   # once, at the start: the key for this process and its children (never stored)
  wandb_on && return 0
  [[ "${WANDB_MODE:-}" == disabled || ! -t 0 || ! -t 1 ]] && return 0
  local k; read -rs -p "    wandb API key (https://wandb.ai/authorize; Enter = no wandb): " k; echo
  if [[ -n "$k" ]]; then export WANDB_API_KEY="$k"; else export WANDB_MODE=disabled; fi
}

wandb_start() {   # wandb_start <run dir> <kind> <gpus> [total steps] [config]
  wandb_on || { info "wandb       off (no key; see z-image/scripts/_wandb.sh)"; return 0; }
  WANDB_RUN_DIR="$1"; mkdir -p "$1/logs"; rm -f "$1/.wandb_stop"
  local args=(--run "$1" --kind "$2" --gpus "$3" --qib "$ROOT/qib/runs" --pid $$)
  [[ -n "${4:-}" ]] && args+=(--total "$4")
  [[ -n "${5:-}" ]] && args+=(--config "$5")
  ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
    cd "$REPO" && exec conda run -n "$ENV_NAME" --no-capture-output python -m zimage_distill.wandb_watch \
      "${args[@]}" ) >> "$1/logs/wandb_watch.log" 2>&1 &
  trap wandb_stop EXIT
  info "wandb       on: ${WANDB_ENTITY:-sw-lmmd}/${WANDB_PROJECT:-zimage-distill}, log $1/logs/wandb_watch.log"
}

wandb_eval() {    # wandb_eval <name> <limit>: this Qwen-Image-Bench run goes to the same wandb run
  [[ -n "$WANDB_RUN_DIR" ]] || return 0
  grep -qs "^$1 " "$WANDB_RUN_DIR/wandb_evals.txt" || echo "$1 ${2:-0}" >> "$WANDB_RUN_DIR/wandb_evals.txt"
}

wandb_stop() {    # on exit: a last pass, then wait (<= 3 min) for the upload to finish
  [[ -n "$WANDB_RUN_DIR" ]] || return 0
  touch "$WANDB_RUN_DIR/.wandb_stop"
  local i; for ((i = 0; i < 180; i++)); do [[ -f "$WANDB_RUN_DIR/.wandb_stop" ]] || break; sleep 1; done
  local url; url=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['url'] or '')" \
                   "$WANDB_RUN_DIR/wandb.json" 2>/dev/null) && [[ -n "$url" ]] && info "wandb: $url"
  WANDB_RUN_DIR=""
}
