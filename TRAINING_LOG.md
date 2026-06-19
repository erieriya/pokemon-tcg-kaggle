# PPO学習runの記録

> `agent/train_ppo.py`を実行するたびにここへ1エントリ追記する。
> `agent/logs/*.log`と`agent/models/*/`は`.gitignore`対象で実行環境のローカルにしか
> 残らないため、「いつ・どんな設定で・どこまで・なぜ止めたか」をこのファイルだけは
> git管理してどのセッションからでも追えるようにしておく。

## ログファイルの命名規則

実行のたびにファイルを上書きしないよう、タイムスタンプ付きで残す。

```
agent/logs/<YYYYMMDD_HHMM>_<run_name>.log
agent/models/<run_name>_<YYYYMMDD_HHMM>/   # チェックポイント保存先
```

`run_experiments.sh`の並行比較実験（`baseline`/`flat_prize`等）は名前自体が設定を
表すスロットなので例外的にこの命名規則の対象外（`agent/logs/<name>.log`に固定で
上書きでよい）。

## 新しいrunを始めるときのコマンド例

```bash
cd agent
RUN=main_2000ep_$(date +%Y%m%d_%H%M)
mkdir -p models/$RUN
nohup uv run python -u train_ppo.py \
  --episodes 2000 \
  --save_dir models/$RUN \
  > logs/${RUN}.log 2>&1 &
disown
echo "launched pid=$!  run=$RUN"
```

`nohup ... & disown`にしておくと、シェルやセッションが切れてもプロセスが残る
（一度`tee`経由でバックグラウンド実行した際にセッション境界でプロセスが
消えてログが空になった事故があったため、この形を標準にする）。

止めるときは `pgrep -f train_ppo.py` でPIDを確認して `kill -TERM <pid>`。

---

## Run履歴

### 2026-06-19 15:04 — `main_2000ep`（テスト実行・途中停止）

- コマンド: `uv run python -u train_ppo.py --episodes 2000 --save_dir models/main_2000ep`
- 設定: デフォルト（opponent_pool: selfplay=0.7, random=0.1, lucario_v1=0.1, lucario_v2=0.1 / prize_rewards=0.05,0.05,0.08,0.1,0.15,0.25 / deckout_win_reward=0.5 / deckout_loss_reward=-1.5 / wipeout_loss_reward=-1.5）
- 結果: EP20まで進行した時点でユーザーの指示により停止。チェックポイントは保存前（save_interval=500未到達）のため`agent/models/main_2000ep/`は空で削除済み。
- ログ: `agent/logs/20260619_1504_main_2000ep_STOPPED_ep20.log`（ローカルのみ、gitignore対象）
- 目的: opponent pool機能（lucario_v1/v2/random/selfplayの重み付きサンプリング）と評価breakdown修正を実装した直後の動作確認。クラッシュなく全対戦相手タイプがサンプリングされることを確認済み（短いスモークテストでも別途確認済み）。本番の長時間学習は次回以降のrunで行う。
- 次にやること: 同じ設定で2000エピソードを本番として再実行する（このrunは検証目的で短時間で止めたものなので、本番 runとしてカウントしない）。
