#!/usr/bin/env bash
# Run judge (accuracy) evaluation from repo root.
# Usage: ./scripts/run_judge.sh [GPU_ID] [MODEL] [MODE] [EXTRA_ARGS...]
# Example: ./scripts/run_judge.sh 0 qwen logit --lam 0.9 --debiasing_target gender
set -e
cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
MODEL="${2:-qwen}"
MODE="${3:-logit}"
shift 3 || true
export CUDA_VISIBLE_DEVICES="$GPU_ID"
python -m tasks.judge.main --model "$MODEL" --mode "$MODE" --gpu_id "$GPU_ID" "$@"
