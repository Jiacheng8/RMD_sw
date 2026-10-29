# Weights & Biases for the launchers (sourced; needs ZDIR, REPO, ROOT, ENV_NAME and info()).
# The logging itself is zimage_distill/wandb_watch.py, a background process that reads the files
# the run writes -- the run never depends on it. On when a key is there: $WANDB_API_KEY, or
# `wandb login` done once on the machine (~/.netrc, any wandb version). The real launchers
# (ours / RDM / DMD2) call wandb_require: they stop before doing anything when there is no login,
# the key is refused, or the account is not in the team -- unless --no-wandb (WANDB_MODE=disabled). Runs go to the team sw-lmmd (WANDB_ENTITY overrides), project zimage-distill
# (WANDB_PROJECT overrides); WANDB_MODE=disabled turns it off; WANDB_TAGS as usual.
WANDB_RUN_DIR=""; WANDB_PID=""

wandb_netrc_key() {   # the key `wandb login` stored: api.wandb.ai (wandb <= 0.30) or
  # api.forge.coreweave.com (wandb >= 0.31, CoreWeave's name for the same service)
  python3 - <<'PY' 2>/dev/null
import netrc, os
try:
    n = netrc.netrc(os.environ.get("NETRC") or os.path.expanduser("~/.netrc"))
except Exception:
    raise SystemExit(1)
for host in ("api.wandb.ai", "api.forge.coreweave.com"):
    a = n.authenticators(host)
    if a and a[2]:
        print(a[2])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

wandb_on() {
  [[ "${WANDB_MODE:-}" == disabled ]] && return 1
  [[ -n "${WANDB_API_KEY:-}" || "${WANDB_MODE:-}" == offline ]] && return 0
  # a login by a newer wandb sits under another host name than the env's wandb 0.30 reads:
  # hand the key over explicitly (it stays in this process and its children)
  local k; k=$(wandb_netrc_key) || return 1
  export WANDB_API_KEY="$k"
}

wandb_ask() {   # once, at the start: the key for this process and its children (never stored)
  wandb_on && return 0
  [[ "${WANDB_MODE:-}" == disabled || ! -t 0 || ! -t 1 ]] && return 0
  local k; read -rs -p "    wandb API key (https://wandb.ai/authorize; Enter = no wandb): " k; echo
  if [[ -n "$k" ]]; then export WANDB_API_KEY="$k"; else export WANDB_MODE=disabled; fi
}

WANDB_REQUIRED=0; WANDB_VERIFIED=0
WANDB_HOWTO="on this machine run: pip install -q wandb && wandb login   (an account in the team ${WANDB_ENTITY:-sw-lmmd}) -- or pass --no-wandb to run without it"

wandb_verify() {  # wandb_verify <python command...>: the key works and the account is in the team
  # rc 0 ok, 1 refused (message on stdout), 2 this python has no wandb (cannot check here)
  "$@" - <<'PY' 2>/dev/null
import os, sys
try:
    import wandb
except ImportError:
    sys.exit(2)
ent = os.environ.get("WANDB_ENTITY") or "sw-lmmd"
try:
    v = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30).viewer
    user = v.username
except Exception as e:
    msg = (str(e).strip().splitlines() or [""])[0][:160]
    print(f"wandb refused the key ({type(e).__name__}: {msg})")
    sys.exit(1)
teams = v.teams
if teams and ent not in teams and ent != user:
    print(f"account {user} is not in the team {ent} (its teams: {', '.join(teams)})")
    sys.exit(1)
print(f"account {user}, team {ent}" if ent in teams else
      f"account {user} (team list not available; runs go to {ent})")
PY
}

wandb_require() { # wandb_require <dry 0|1>: the real launchers refuse to start without a working login
  local dry="${1:-0}" out rc k
  if [[ "${WANDB_MODE:-}" == disabled ]]; then info "wandb       off (--no-wandb / WANDB_MODE=disabled)"; return 0; fi
  WANDB_REQUIRED=1
  if ! wandb_on && [[ -t 0 && -t 1 && "$dry" == 0 ]]; then
    read -rs -p "    wandb API key (https://wandb.ai/authorize): " k; echo
    [[ -n "$k" ]] && export WANDB_API_KEY="$k"
  fi
  if ! wandb_on; then
    [[ "$dry" == 1 ]] && { info "wandb       MISSING -- the real run will stop here; $WANDB_HOWTO"; return 0; }
    die "no wandb login found. $WANDB_HOWTO"
  fi
  [[ "${WANDB_MODE:-}" == offline ]] && { info "wandb       offline"; WANDB_VERIFIED=1; return 0; }
  out=$(wandb_verify python3) && rc=0 || rc=$?
  if (( rc == 2 )) && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    out=$(wandb_verify conda run -n "$ENV_NAME" --no-capture-output python) && rc=0 || rc=$?
  fi
  case $rc in
    0) info "wandb       login ok: $out"; WANDB_VERIFIED=1 ;;
    2) info "wandb       key found; checked once the env is installed (before training)" ;;
    *) [[ "$dry" == 1 ]] && { info "wandb       FAILS -- $out; the real run will stop here"; return 0; }
       die "$out. $WANDB_HOWTO" ;;
  esac
}

wandb_start() {   # wandb_start <run dir> <kind> <gpus> [total steps] [config]
  if (( WANDB_REQUIRED && ! WANDB_VERIFIED )); then
    local out
    wandb_on || die "no wandb login found. $WANDB_HOWTO"
    out=$(wandb_verify conda run -n "$ENV_NAME" --no-capture-output python) \
      || die "${out:-could not check the wandb login}. $WANDB_HOWTO"
    info "wandb       login ok: $out"; WANDB_VERIFIED=1
  fi
  wandb_on || { info "wandb       off (no key; see z-image/scripts/_wandb.sh)"; return 0; }
  WANDB_RUN_DIR="$1"; mkdir -p "$1/logs"; rm -f "$1/.wandb_stop"
  local args=(--run "$1" --kind "$2" --qib "$ROOT/qib/runs" --pid $$)
  [[ -n "${3:-}" ]] && args+=(--gpus "$3")
  [[ -n "${4:-}" ]] && args+=(--total "$4")
  [[ -n "${5:-}" ]] && args+=(--config "$5")
  ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
    cd "$REPO" && exec conda run -n "$ENV_NAME" --no-capture-output python -m zimage_distill.wandb_watch \
      "${args[@]}" ) >> "$1/logs/wandb_watch.log" 2>&1 &
  WANDB_PID=$!
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
  local i; for ((i = 0; i < 180; i++)); do kill -0 "$WANDB_PID" 2>/dev/null || break; sleep 1; done
  rm -f "$WANDB_RUN_DIR/.wandb_stop"
  grep -h "^\[wandb\] not logging" "$WANDB_RUN_DIR/logs/wandb_watch.log" 2>/dev/null | tail -1 | sed "s/^/    /" || true
  local url; url=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['url'] or '')" \
                   "$WANDB_RUN_DIR/wandb.json" 2>/dev/null) && [[ -n "$url" ]] && info "wandb: $url"
  WANDB_RUN_DIR=""
}
