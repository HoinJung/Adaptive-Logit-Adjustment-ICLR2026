#!/usr/bin/env bash
# Run counterfactual (toxicity) debiasing from repo root.
# Usage: ./scripts/run_counterfactual.sh [GPU_ID] [MODEL] [MODE] [EXTRA_ARGS...]
# Example: ./scripts/run_counterfactual.sh 0 qwen logit --lam 1.0 --target gender
set -e
cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
MODEL="${2:-qwen}"
MODE="${3:-naive}"
shift 3 || true
export CUDA_VISIBLE_DEVICES="$GPU_ID"
python -m tasks.counterfactual.main --model "$MODEL" --mode "$MODE" --gpu_id "$GPU_ID" "$@"
