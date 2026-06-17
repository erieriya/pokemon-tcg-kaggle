#!/bin/bash
# 提出スクリプト: ./submit.sh "コメント"
#
# 使い方:
#   ./submit.sh "heuristic v3"
#   ./submit.sh "PPO 500k steps"

set -e

DESCRIPTION="${1:-no description}"
COMPETITION="pokemon-tcg-ai-battle"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. submission/ ディレクトリを再構成
echo "[1/3] Preparing submission/ ..."
mkdir -p "$SCRIPT_DIR/submission"
cp "$SCRIPT_DIR/agent/dragapult_agent.py" "$SCRIPT_DIR/submission/main.py"
cp "$SCRIPT_DIR/agent/deck.csv"           "$SCRIPT_DIR/submission/deck.csv"

# cg/ を毎回 data/sample_submission から新鮮にコピー
rm -rf "$SCRIPT_DIR/submission/cg"
cp -r  "$SCRIPT_DIR/data/sample_submission/cg" "$SCRIPT_DIR/submission/cg"
echo "      cg/ files: $(ls submission/cg/ | tr '\n' ' ')"

# 2. tar.gz を生成
echo "[2/3] Building submission.tar.gz ..."
cd "$SCRIPT_DIR/submission"
tar -czf "$SCRIPT_DIR/submission.tar.gz" main.py deck.csv cg/
cd "$SCRIPT_DIR"
echo "      $(du -sh submission.tar.gz | cut -f1)"

# 3. Kaggle CLI で submit
echo "[3/3] Submitting to Kaggle ..."
kaggle competitions submit -c "$COMPETITION" -f submission.tar.gz -m "$DESCRIPTION"

echo ""
echo "Done! Check status:"
echo "  kaggle competitions submissions -c $COMPETITION"
