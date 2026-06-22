# Repository Guidelines

## プロジェクト概要
Kaggleコンペ「pokemon-tcg-ai-battle」向けのポケモンカード対戦AI開発リポジトリ。
ヒューリスティック型エージェントとPPOで学習する強化学習(RL)エージェントの両系統を持つ。

## ディレクトリ構成
- `agent/` — エージェント実装の本体
  - `<デッキ名>_agent.py` + `deck_<デッキ名>.csv` がペアで存在(例: `dragapult_agent.py` / `deck_dragapult.csv`)
  - `train_ppo.py` — PPO自己対戦学習スクリプト
  - `rl_agent.py` / `rl_main.py` — 学習済みモデルを読み込む推論用エージェント
  - `main.py` — 提出用エントリーポイント。環境変数 `PTCG_AGENT`(dragapult/heuristic/random/rl) で切り替え
  - `models/` (gitignore対象) — 学習チェックポイント、`logs/` (gitignore対象) — 学習ログ
- `tools/` — `run_battle.py`(対戦実行)、`visualize_battle.py` / `ptcg_viewer.py`(可視化)、`test_agent.py`
- `data/` — カードデータ(`EN_Card_Data.csv` / `JP_Card_Data.csv`)、`sample_submission/cg/`(提出に必須のコアゲームライブラリ)
- `battle_logs/` / `battle_replays/` (gitignore対象) — 対戦結果・再生データの出力先
- `submission/` / `submission.tar.gz` (gitignore対象) — 提出物のビルド先
- `TRAINING_LOG.md` / `REWARD_IDEAS.md` / `REWARD_CHANGES.md` / `research.md` — 学習・報酬設計の知見の記録(変更時はここに追記する運用)

## 環境構築・実行コマンド
- 依存管理は `uv`(`pyproject.toml` / `uv.lock`、Python 3.11)。スクリプトは `uv run python ...` で実行する
- 単発のPPO学習: `cd agent && uv run python train_ppo.py --episodes <N> --prize_rewards <...> --deckout_win_reward <...> --deckout_loss_reward <...> --wipeout_loss_reward <...>`
- 複数の報酬設定を並列で学習: `./run_experiments.sh [episodes]`(設定は内部の `EXPERIMENTS` 配列を編集)
- 対戦の実行・可視化: `uv run python tools/run_battle.py` / `uv run python tools/visualize_battle.py`

## 提出フロー
- ヒューリスティック系: `./submit.sh "コメント"` → `agent/dragapult_agent.py` + `agent/deck.csv` + `data/sample_submission/cg/` を `submission/` にまとめて `submission.tar.gz` を作成し `kaggle competitions submit` する
- RL系: `./submit_rl.sh "コメント" [モデルパス(省略時 agent/models/model_final.pt)]` → `rl_main.py` + `rl_agent.py` + 指定モデル + `deck.csv` + `cg/` をまとめて提出
- `kaggle_api_key.txt` / `kaggle.json` は秘匿情報。読み取り・コミット・ログ出力に含めない

## コーディング規約
- 新しいデッキ用エージェントを追加する場合は既存の `<デッキ名>_agent.py` / `deck_<デッキ名>.csv` の命名・配置パターンに従う
- 大きいデータファイル(カード画像、PDF、tar.gz、学習済みモデル等)は `.gitignore` に従いコミットしない

## 制約・注意事項
- `data/card_images/`、`data/Card_ID List_JP.pdf` 等の大容量ファイルはgit管理外
- 報酬設計やハイパーパラメータを変更した場合は `TRAINING_LOG.md` / `REWARD_CHANGES.md` に変更内容と結果を記録する
