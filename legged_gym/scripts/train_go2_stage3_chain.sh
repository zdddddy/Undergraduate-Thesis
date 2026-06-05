#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-}"

cd "$ROOT_DIR"
if [[ -n "$CONDA_BIN_DIR" ]]; then
  export PATH="$CONDA_BIN_DIR:$PATH"
fi
export SIMULATOR="isaacgym"

# Required: Stage2b run directory for warm-starting stage2c.
# Example:
#   export STAGE2B_RUN="$ROOT_DIR/results/training_logs/go2_stage2/Apr11_15-47-04_stage2b_isaacgym"
STAGE2B_RUN="${STAGE2B_RUN:-}"
if [[ -z "$STAGE2B_RUN" ]]; then
  echo "[ERROR] STAGE2B_RUN is required."
  echo "Set it to a Stage2b run directory, then rerun."
  exit 1
fi
if [[ ! -d "$STAGE2B_RUN" ]]; then
  echo "[ERROR] STAGE2B_RUN not found: $STAGE2B_RUN"
  exit 1
fi

STAGE2B_CKPT="${STAGE2B_CKPT:--1}"

STAGE3_ITERS="${STAGE3_ITERS:-5000}"

CHAIN_LOG_DIR="$ROOT_DIR/results/chain_logs/go2_stage3"
mkdir -p "$CHAIN_LOG_DIR"
CHAIN_TS="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$CHAIN_LOG_DIR/${CHAIN_TS}.log"

{
  echo "[$(date '+%F %T')] Stage2b->2c->3 chain start"
  echo "[$(date '+%F %T')] ROOT_DIR=$ROOT_DIR"
  echo "[$(date '+%F %T')] SIMULATOR=$SIMULATOR"
  echo "[$(date '+%F %T')] PYTHON=$PYTHON_BIN"
  echo "[$(date '+%F %T')] STAGE2B_RUN=$STAGE2B_RUN"
  echo "[$(date '+%F %T')] STAGE2B_CKPT=$STAGE2B_CKPT"
  echo "[$(date '+%F %T')] STAGE3_ITERS=$STAGE3_ITERS"

  echo "[$(date '+%F %T')] Launching 2c: go2_stage2c --resume --load_run $STAGE2B_RUN --ckpt $STAGE2B_CKPT"
  "$PYTHON_BIN" -m legged_gym.scripts.train \
    --task go2_stage2c \
    --resume \
    --load_run "$STAGE2B_RUN" \
    --ckpt "$STAGE2B_CKPT" \
    --headless

  RUN_ROOT="$ROOT_DIR/results/training_logs/go2_stage2"
  LOAD_RUN_PATH="$(ls -dt "$RUN_ROOT"/*stage2c* 2>/dev/null | head -n1 || true)"
  if [[ -z "$LOAD_RUN_PATH" ]]; then
    echo "[$(date '+%F %T')] ERROR: cannot find stage2c run under $RUN_ROOT"
    exit 1
  fi
  echo "[$(date '+%F %T')] 2c finished. Resume source for stage3: $LOAD_RUN_PATH"

  echo "[$(date '+%F %T')] Launching 3: go2_stage3 --resume --load_run $LOAD_RUN_PATH --ckpt -1 --max_iterations $STAGE3_ITERS"
  "$PYTHON_BIN" -m legged_gym.scripts.train \
    --task go2_stage3 \
    --resume \
    --load_run "$LOAD_RUN_PATH" \
    --ckpt -1 \
    --max_iterations "$STAGE3_ITERS" \
    --headless

  echo "[$(date '+%F %T')] Stage2b->2c->3 chain completed"
} 2>&1 | tee -a "$CHAIN_LOG"
