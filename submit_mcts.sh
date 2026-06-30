#!/bin/bash
# MCTS エージェントの提出スクリプト: ./submit_mcts.sh "コメント" [モデルパス]

set -e

DESCRIPTION="${1:-mcts agent}"
MODEL_PATH="${2:-agent/models/mcts_loop_v5.pt}"
COMPETITION="pokemon-tcg-ai-battle"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] Preparing submission/ ..."
mkdir -p "$SCRIPT_DIR/submission"
cp "$SCRIPT_DIR/agent/mcts_main.py" "$SCRIPT_DIR/submission/main.py"
cp "$SCRIPT_DIR/agent/rl_agent.py"  "$SCRIPT_DIR/submission/rl_agent.py"
cp "$SCRIPT_DIR/agent/mcts.py"      "$SCRIPT_DIR/submission/mcts.py"
cp "$SCRIPT_DIR/agent/train_ppo.py" "$SCRIPT_DIR/submission/train_ppo.py"
cp "$SCRIPT_DIR/agent/deck.csv"     "$SCRIPT_DIR/submission/deck.csv"
cp "$SCRIPT_DIR/$MODEL_PATH"        "$SCRIPT_DIR/submission/model_final.pt"

rm -rf "$SCRIPT_DIR/submission/cg"
cp -r  "$SCRIPT_DIR/data/sample_submission/cg" "$SCRIPT_DIR/submission/cg"
rm -rf "$SCRIPT_DIR/submission/cg/__pycache__"
echo "      model: $MODEL_PATH ($(du -sh "$SCRIPT_DIR/$MODEL_PATH" | cut -f1))"

echo "[2/3] Building submission.tar.gz ..."
cd "$SCRIPT_DIR/submission"
tar -czf "$SCRIPT_DIR/submission.tar.gz" main.py rl_agent.py mcts.py train_ppo.py model_final.pt deck.csv cg/
cd "$SCRIPT_DIR"
echo "      $(du -sh submission.tar.gz | cut -f1)"

echo "[3/3] Submitting to Kaggle ..."
kaggle competitions submit -c "$COMPETITION" -f submission.tar.gz -m "$DESCRIPTION"

echo ""
echo "Done! Check status:"
echo "  kaggle competitions submissions -c $COMPETITION"
