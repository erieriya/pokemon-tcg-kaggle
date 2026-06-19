#!/bin/bash
# 異なる報酬設定で複数のPPO自己対戦学習を同時に走らせる。
#
# 使い方:
#   ./run_experiments.sh [episodes]
#   ./run_experiments.sh 2000
#
# 設定を変えたいときはEXPERIMENTS配列を編集する。
# 各行: "名前|サイド1〜6枚目の中間報酬(カンマ区切り)|デッキアウト勝ちの報酬|デッキアウト負けの罰則|全滅負けの罰則"
#
# 各設定は agent/models/<名前>/ にチェックポイントを保存し、
# agent/logs/<名前>.log に標準出力を書き出す。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EPISODES="${1:-2000}"
LOG_DIR="$SCRIPT_DIR/agent/logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  "baseline|0.05,0.05,0.08,0.1,0.15,0.25|0.5|-1.0|-1.0"
  "flat_prize|0.1,0.1,0.1,0.1,0.1,0.1|0.5|-1.0|-1.0"
  "late_heavy|0.02,0.02,0.04,0.08,0.2,0.4|0.5|-1.0|-1.0"
  "penalize_passive|0.05,0.05,0.08,0.1,0.15,0.25|0.2|-1.5|-1.5"
)

PIDS=()
for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name prize_rewards deckout_win_reward deckout_loss_reward wipeout_loss_reward <<< "$entry"
  save_dir="$SCRIPT_DIR/agent/models/$name"
  mkdir -p "$save_dir"
  echo "[launch] $name: prize_rewards=$prize_rewards deckout_win_reward=$deckout_win_reward deckout_loss_reward=$deckout_loss_reward wipeout_loss_reward=$wipeout_loss_reward -> $save_dir"
  (
    cd "$SCRIPT_DIR/agent" && \
    uv run python train_ppo.py \
      --episodes "$EPISODES" \
      --prize_rewards "$prize_rewards" \
      --deckout_win_reward "$deckout_win_reward" \
      --deckout_loss_reward "$deckout_loss_reward" \
      --wipeout_loss_reward "$wipeout_loss_reward" \
      --save_dir "$save_dir"
  ) > "$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
done

echo ""
echo "Launched ${#PIDS[@]} training runs: ${PIDS[*]}"
echo "Logs:    tail -f $LOG_DIR/*.log"
echo "Stop all: kill ${PIDS[*]}"
