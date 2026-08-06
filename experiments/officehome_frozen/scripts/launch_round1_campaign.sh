#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_ROOT="$REPO_ROOT/experiments/officehome_frozen"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/results/domain_adaptation/officehome_frozen/campaigns/round1_150q}"
STAGE="${STAGE:-full}"
TMUX_SESSION="${TMUX_SESSION:-officehome_round1_150q}"
PYTHON="${PYTHON:-$EXP_ROOT/.venv/bin/python}"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_SESSION" >&2
  exit 2
fi

if [[ -z "${PHYSICAL_GPU:-}" ]]; then
  PHYSICAL_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi

mkdir -p "$CAMPAIGN_ROOT/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$CAMPAIGN_ROOT/logs/${STAGE}_${STAMP}.log"
PID_PATH="$CAMPAIGN_ROOT/${STAGE}.pid"

COMMAND="cd '$REPO_ROOT' && echo \$\$ > '$PID_PATH' && \
env CUDA_VISIBLE_DEVICES='$PHYSICAL_GPU' PYTHONUNBUFFERED=1 '$PYTHON' \
'$EXP_ROOT/scripts/run_officehome_campaign.py' --stage '$STAGE' --device auto \
--campaign-root '$CAMPAIGN_ROOT' 2>&1 | tee -a '$LOG_PATH'"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc \"set -o pipefail; $COMMAND\""

echo "tmux_session=$TMUX_SESSION"
echo "physical_gpu=$PHYSICAL_GPU"
echo "log_path=$LOG_PATH"
echo "pid_path=$PID_PATH"
echo "follow=tmux capture-pane -pt $TMUX_SESSION -S -80"
echo "log_follow=tail -f $LOG_PATH"
