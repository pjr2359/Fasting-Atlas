#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GOLD="${GOLD:-$ROOT/scripts/eval/gold_template.csv}"
PARSED="${1:-$ROOT/parsed}"
OUT="${2:-$ROOT/eval_report.json}"
exec "$ROOT/.venv/bin/python" -m fasting_atlas.cli eval --gold "$GOLD" --parsed "$PARSED" --out "$OUT"
