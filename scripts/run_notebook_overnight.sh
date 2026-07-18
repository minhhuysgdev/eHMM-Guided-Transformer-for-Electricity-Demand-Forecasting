#!/usr/bin/env bash
# Chạy notebook headless qua đêm (không cần mở Cursor/Jupyter UI).
# Giữ Mac thức bằng caffeinate — nên cắm sạc.
#
# Usage:
#   ./scripts/run_notebook_overnight.sh
#   ./scripts/run_notebook_overnight.sh tmux    # chạy trong tmux, detach được
#
# Theo dõi log:
#   tail -f logs/notebook_*.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOK="${NOTEBOOK:-$ROOT/notebooks/01_ehmm_transformer_implementation.ipynb}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/notebook_${TIMESTAMP}.log"
OUT_NB="$LOG_DIR/01_executed_${TIMESTAMP}.ipynb"
PID_FILE="$LOG_DIR/notebook_${TIMESTAMP}.pid"

run_notebook() {
  cd "$ROOT"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"

  echo "=== Start: $(date) ===" | tee -a "$LOG_FILE"
  echo "Notebook: $NOTEBOOK" | tee -a "$LOG_FILE"
  echo "Output:   $OUT_NB" | tee -a "$LOG_FILE"
  echo "Device check:" | tee -a "$LOG_FILE"
  python -c "import torch; print('torch', torch.__version__, '| mps:', torch.backends.mps.is_available())" | tee -a "$LOG_FILE"

  # timeout=-1: không giới hạn thời gian cell (train có thể vài giờ)
  caffeinate -dimsu python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$OUT_NB" \
    "$NOTEBOOK" \
    2>&1 | tee -a "$LOG_FILE"

  echo "=== Done: $(date) ===" | tee -a "$LOG_FILE"
}

if [[ "${1:-}" == "tmux" ]]; then
  SESSION="ehmm_train"
  tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
  tmux new-session -d -s "$SESSION" "bash '$0' 2>&1 | tee '$LOG_FILE'; echo 'Finished. Press Enter.'; read"
  echo "tmux session: $SESSION"
  echo "Attach:  tmux attach -t $SESSION"
  echo "Log:     tail -f '$LOG_FILE'"
  exit 0
fi

echo $$ > "$PID_FILE"
run_notebook
``