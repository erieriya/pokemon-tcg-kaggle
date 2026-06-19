#!/bin/bash
# 学習済みRLエージェントの提出スクリプト: ./submit_rl.sh "コメント" [モデルパス]
#
# 使い方:
#   ./submit_rl.sh "PPO 2000ep self-play"
#   ./submit_rl.sh "PPO 2000ep self-play" agent/models/model_ep750.pt

set -e

DESCRIPTION="${1:-no description}"
MODEL_PATH="${2:-agent/models/model_final.pt}"
COMPETITION="pokemon-tcg-ai-battle"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. submission/ ディレクトリを再構成
echo "[1/3] Preparing submission/ ..."
mkdir -p "$SCRIPT_DIR/submission"
cp "$SCRIPT_DIR/agent/rl_main.py"  "$SCRIPT_DIR/submission/main.py"
cp "$SCRIPT_DIR/agent/rl_agent.py" "$SCRIPT_DIR/submission/rl_agent.py"
cp "$SCRIPT_DIR/agent/deck.csv"    "$SCRIPT_DIR/submission/deck.csv"
cp "$SCRIPT_DIR/$MODEL_PATH"       "$SCRIPT_DIR/submission/model_final.pt"

# cg/ を毎回 data/sample_submission から新鮮にコピー (__pycache__ 除外)
rm -rf "$SCRIPT_DIR/submission/cg"
cp -r  "$SCRIPT_DIR/data/sample_submission/cg" "$SCRIPT_DIR/submission/cg"
rm -rf "$SCRIPT_DIR/submission/cg/__pycache__"
echo "      cg/ files: $(ls "$SCRIPT_DIR/submission/cg" | tr '\n' ' ')"
echo "      model: $MODEL_PATH"

# 2. tar.gz を生成
echo "[2/3] Building submission.tar.gz ..."
cd "$SCRIPT_DIR/submission"
tar -czf "$SCRIPT_DIR/submission.tar.gz" main.py rl_agent.py model_final.pt deck.csv cg/
cd "$SCRIPT_DIR"
echo "      $(du -sh submission.tar.gz | cut -f1)"

# 3. Kaggle CLI で submit
echo "[3/3] Submitting to Kaggle ..."
kaggle competitions submit -c "$COMPETITION" -f submission.tar.gz -m "$DESCRIPTION"

echo ""
echo "Done! Check status:"
echo "  kaggle competitions submissions -c $COMPETITION"
