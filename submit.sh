#!/bin/bash
# 提出スクリプト: ./submit.sh "コメント"
#
# 使い方:
#   ./submit.sh "heuristic v1"
#   ./submit.sh "PPO 500k steps"

set -e

DESCRIPTION="${1:-no description}"
TOKEN=$(cat "$(dirname "$0")/kaggle_api_key.txt")
COMPETITION="pokemon-tcg-ai-battle"

# 1. tar.gz を再生成
echo "[1/2] Building submission.tar.gz ..."
cp agent/dragapult_agent.py submission/main.py
cp agent/deck.csv submission/deck.csv
cd submission && tar -czf ../submission.tar.gz main.py deck.csv cg/ && cd ..
echo "      $(du -sh submission.tar.gz | cut -f1)"

# 2. Kaggle に submit
echo "[2/2] Submitting to Kaggle ..."
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "competition=$COMPETITION" \
  -F "submissionDescription=$DESCRIPTION" \
  -F "file=@submission.tar.gz" \
  "https://www.kaggle.com/api/v1/competitions/$COMPETITION/submissions" \
  | python3 -m json.tool

echo ""
echo "Done! Check status at: https://www.kaggle.com/competitions/$COMPETITION/submissions"
