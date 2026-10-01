#!/usr/bin/env bash
# Score a GenEval render dir with the canonical scorer (docs/geneval_protocol.md), set up by
# scripts/setup_geneval.sh.
#
#   bash scripts/score_geneval.sh <render_dir> --root <geneval-root> [--out <results.jsonl>]
#
# <render_dir>/<idx:05d>/{metadata.jsonl, samples/NNNN.png} -- what rdm's render_geneval writes
# (reproduce.py eval-flux -> <output_dir>/geneval). Writes <out> (default <render_dir>/../
# geneval_results.jsonl), prints upstream's summary_scores.py table, and writes <out>.summary.json
# whose "overall" is the canonical GenEval score: the UNWEIGHTED mean of the 6 task rates.
# Pick the GPU with CUDA_VISIBLE_DEVICES.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDER=""
ROOT=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --out)  OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)  RENDER="$1"; shift ;;
  esac
done
[[ -n "$RENDER" && -n "$ROOT" ]] || { echo "usage: bash scripts/score_geneval.sh <render_dir> --root <geneval-root> [--out FILE]" >&2; exit 2; }
RENDER="$(realpath "$RENDER")"
ROOT="$(realpath "$ROOT")"
[[ -f "$ROOT/geneval_env.sh" ]] || { echo "$ROOT/geneval_env.sh not found -- run scripts/setup_geneval.sh --root $ROOT first" >&2; exit 1; }
source "$ROOT/geneval_env.sh"
OUT="$(realpath -m "${OUT:-$(dirname "$RENDER")/geneval_results.jsonl}")"
N_PROMPTS=$(find -L "$RENDER" -mindepth 2 -maxdepth 2 -name metadata.jsonl | wc -l)
N_IMAGES=$(find -L "$RENDER" -mindepth 3 -maxdepth 3 -path '*/samples/*' -name '[0-9]*.png' | wc -l)
[[ $N_IMAGES -gt 0 ]] || { echo "no <idx>/samples/NNNN.png images under $RENDER" >&2; exit 1; }
echo "==> GenEval scoring $N_IMAGES images / $N_PROMPTS prompts in $RENDER"

# open_clip fetches the ViT-L-14 (openai) colour classifier from the HF hub on first use.
export HF_HOME="${HF_HOME:-$GENEVAL_HF_HOME}"
CRUN=(conda run "${GENEVAL_ENV_ARGS[@]}" --no-capture-output)
"${CRUN[@]}" python "$GENEVAL_SRC/evaluation/evaluate_images.py" "$RENDER" \
    --outfile "$OUT" --model-path "$GENEVAL_MODELS" --model-config "$GENEVAL_MMDET_CONFIG" \
    --options model=mask2former
N_ROWS=$(wc -l < "$OUT")
[[ $N_ROWS -eq $N_IMAGES ]] || { echo "scorer wrote $N_ROWS rows for $N_IMAGES images" >&2; exit 1; }
"${CRUN[@]}" python "$GENEVAL_SRC/evaluation/summary_scores.py" "$OUT"
"${CRUN[@]}" python - "$OUT" <<'PY'
import json, sys
import pandas as pd
out = sys.argv[1]
df = pd.read_json(out, orient="records", lines=True)
tasks = {tag: float(g["correct"].mean()) for tag, g in df.groupby("tag", sort=False)}
summary = {**tasks, "overall": sum(tasks.values()) / len(tasks), "images": int(len(df))}
json.dump(summary, open(out + ".summary.json", "w"), indent=2)
print(f"==> overall {summary['overall']:.4f}  ->  {out}.summary.json")
PY
