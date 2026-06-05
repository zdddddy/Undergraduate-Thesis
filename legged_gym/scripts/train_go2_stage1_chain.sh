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

CHAIN_LOG_DIR="$ROOT_DIR/results/chain_logs/go2_stage1"
mkdir -p "$CHAIN_LOG_DIR"
CHAIN_TS="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$CHAIN_LOG_DIR/${CHAIN_TS}.log"

{
  echo "[$(date '+%F %T')] Stage1 chain start"
  echo "[$(date '+%F %T')] ROOT_DIR=$ROOT_DIR"
  echo "[$(date '+%F %T')] SIMULATOR=$SIMULATOR"
  echo "[$(date '+%F %T')] PYTHON=$PYTHON_BIN"

  echo "[$(date '+%F %T')] Launching 1a: go2_stage1_1a"
  "$PYTHON_BIN" -m legged_gym.scripts.train --task go2_stage1_1a --headless

  RUN_ROOT="$ROOT_DIR/results/training_logs/go2_stage1"
  LOAD_RUN_PATH="$(ls -dt "$RUN_ROOT"/*stage1_1a* 2>/dev/null | head -n1 || true)"
  if [[ -z "$LOAD_RUN_PATH" ]]; then
    echo "[$(date '+%F %T')] ERROR: cannot find stage1_1a run under $RUN_ROOT"
    exit 1
  fi
  LOAD_RUN_NAME="$(basename "$LOAD_RUN_PATH")"
  echo "[$(date '+%F %T')] 1a finished. Resume source for 1b: $LOAD_RUN_NAME"

  echo "[$(date '+%F %T')] Launching 1b: go2_stage1_1b --resume --load_run $LOAD_RUN_NAME --ckpt -1"
  "$PYTHON_BIN" -m legged_gym.scripts.train --task go2_stage1_1b --resume --load_run "$LOAD_RUN_NAME" --ckpt -1 --headless

  echo "[$(date '+%F %T')] Stage1 chain completed"
} 2>&1 | tee -a "$CHAIN_LOG"
