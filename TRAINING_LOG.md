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

### 2026-06-19 15:09 — `main_2000ep_20260619_1509`（本番実行・進行中）

- コマンド: `uv run python -u train_ppo.py --episodes 2000 --save_dir models/main_2000ep_20260619_1509`
- 設定: 上記テスト実行と同じデフォルト設定（opponent_pool: selfplay=0.7, random=0.1, lucario_v1=0.1, lucario_v2=0.1 / prize_rewards=0.05,0.05,0.08,0.1,0.15,0.25 / deckout_win_reward=0.5 / deckout_loss_reward=-1.5 / wipeout_loss_reward=-1.5）。eval_interval/eval_games/save_intervalもデフォルト（それぞれ200/10/500）。
- 起動方法: `nohup ... & disown`（セッション境界で死なないようにするため）。
- ログ: `agent/logs/main_2000ep_20260619_1509.log`（ローカルのみ）
- チェックポイント: `agent/models/main_2000ep_20260619_1509/`にepisode 500/1000/1500/2000(final)で保存される予定。
- 状態: **完了**(2000エピソード、`model_final.pt`保存済み)。
- 最終評価(20戦ずつ、`agent/`から`train_ppo.evaluate()`を直接呼んで測定): random=80.0%, lucario_v1=0.0%, lucario_v2=0.0%, crustle=0.0%, iono=0.0%, abomasnow=0.0%。
- 考察: ランダムには勝てるが、ルールベースのヒューリスティック対戦相手には20戦全敗。self-play比重70%・heuristic比重が低い(各10%以下)ため、Dragapult exミラー以外の戦い方(特にCrustleウォール対策)を学習する機会が少なすぎたと判断。次のrunで対戦相手の重みを見直す(下記参照)。

### 2026-06-19 18:49 — `rebalanced_2000ep_20260619_1849` / `rebalanced_20000ep_20260619_1849`（本番実行・進行中）

上記`main_2000ep_20260619_1509`がheuristic対戦相手に全敗だった反省を受けて、対戦相手の重みをself-play寄りから固定対戦相手寄りに変更。同じ条件で2000エピソード版と20000エピソード版を同時に走らせ、エピソード数を増やすだけで改善するか/対戦相手配分を変える方が効くかを切り分ける狙い。reward設計は今回は変更しない([REWARD_IDEAS.md](REWARD_IDEAS.md)で保留中のまま)。

- コマンド(両run共通、`--episodes`のみ違う):
  ```
  uv run python -u train_ppo.py --episodes <2000|20000> \
    --selfplay_weight 0.35 --crustle_weight 0.25 \
    --lucario_v1_weight 0.1 --lucario_v2_weight 0.1 --iono_weight 0.1 --abomasnow_weight 0.1 \
    --random_weight 0 \
    --save_dir models/<run名>
  ```
- 設定変更点: selfplay 0.7→0.35、random 0.1→0(除外)、crustle 0.1→0.25(最重要メタなので最大配分)、lucario_v1/v2/iono/abomasnowは0.05→0.1に統一。reward関連パラメータ(prize_rewards等)は前回と同じデフォルト。
- 起動方法: `nohup ... & disown`、2プロセス同時実行(GPU/CPUとも余裕ありを確認済み: RTX4090 24GB中41MB使用、32コア中通常1〜2コア専有)。
- ログ: `agent/logs/rebalanced_2000ep_20260619_1849.log` / `agent/logs/rebalanced_20000ep_20260619_1849.log`(ローカルのみ)
- チェックポイント: `agent/models/rebalanced_2000ep_20260619_1849/` / `agent/models/rebalanced_20000ep_20260619_1849/`
- 状態: `rebalanced_2000ep_20260619_1849`は**完了**。20戦評価: lucario_v1=0.0%, lucario_v2=0.0%, crustle=0.0%, iono=0.0%, abomasnow=10.0%(旧重みの2000ep版とほぼ同水準。重みを変えただけでは2000エピソードでは差が出なかった)。

#### `rebalanced_20000ep_20260619_1849`で発生したクラッシュとバグ修正(2026-06-20)

EP5340付近で2回連続でクラッシュした(`battle_select`が`IndexError`)。原因は2つの独立したバグ:

1. `agent/train_ppo.py`の`PPOTrainer.select_action`と`play_eval_game`に`select.get("maxCount", 1) or 1`という記述があり、`maxCount`/`minCount`が正当に`0`(強制選択なしの場面)のときも`or 1`で`1`に書き換えてしまい、エンジンに拒否されていた。→ `or 1`を削除。
2. `agent/lucario_v1_agent.py`(Kaggle notebookからの移植元コードに同じバグがあった)の`k = max(k, min(max(1, select.minCount), n))`と`_legal_fallback`内の`max(1, select.minCount)`も同種のバグ。→ 同様に`max(1, ...)`を削除。

合わせて、対戦相手(opponent_fn)の返り値を検証する安全策が型・範囲チェックのみで「個数(minCount/maxCount)」「重複の有無」を見ていなかったため、新しい共通ヘルパー`_sanitize_opponent_action`(`agent/train_ppo.py`)/`_sanitize_action`(`tools/run_battle.py`)を追加し、不正な返り値は確実に合法手へフォールバックするようにした(将来似たバグが他のヒューリスティックに見つかっても学習プロセス自体は落ちないようにする狙い)。

修正後`model_ep5000.pt`から`--resume`したが、**EP5420付近で3回目のクラッシュ**(今度はself-play側`play_episode`内)が発生。原因は未特定(上記2つの既知バグとは別経路)。原因調査より学習runを安定して進めることを優先し、`battle_select`の`IndexError`を**1エピソード単位で捨てて学習run自体は継続する**耐性を追加:

- `agent/train_ppo.py`の`play_episode`/`play_episode_vs_opponent`に`try/except IndexError`を追加。発生時は該当エピソード分だけ`trainer.buffers`を巻き戻し(GAE計算を汚さないため)、`battle_finish()`で後始末し、`info["aborted"]=True`を返す。
- `train()`側は`aborted`なエピソードを統計に数えず1試合分捨てて次へ進む。20エピソード連続でabortしたら(未知の系統的な問題の可能性が高いため)`RuntimeError`で停止する。
- 発生時の詳細(`SelectContext`/min・maxCount/option種別/実際のaction)を`agent/logs/crash_diagnostics.jsonl`に追記するようにした。次回似た現象が起きたらこのログで原因を特定する。

`model_ep5000.pt`から再度`--resume`済み(2026-06-22 15:51〜)。クラッシュ診断から判明した実際の原因: `_sequential_sample`/`_sequential_log_prob`が重複除外に`logits[idx] = -1e9`という大きな負の定数を使っていたが、学習が進んでlogitsの絶対値がこの定数に対して十分小さくない場面で、既に選んだindexを再度サンプリングしてしまっていた(crash_diagnostics.jsonlで`action: [X, X]`という完全重複が多数確認できた)。booleanマスク+`-inf`で確率を構造的に0にする実装に修正(400エピソードのスモークテストでabort 0件を確認)。

**2026-06-22 17:xx — ユーザー指示により学習を一時停止。**
新しい状態/行動エンコーダ(`agent/rl_agent.py`に特徴量4種を追加: ATTACKの弱点/抵抗補正後実効ダメージ+KO可能フラグ、相手ベンチのスカラー特徴、スタジアムカード特徴、相手の捨て札プール埋め込み)を別タスク(Codex CLI)で追加検討中のため、`rebalanced_20000ep_20260619_1849`はEP5000のチェックポイントで一旦停止。

- 既存チェックポイント(`model_ep500`〜`model_ep5000.pt`、`main_2000ep`/`rebalanced_2000ep`の`model_final.pt`等)は全て**旧エンコーダ(入力次元776)**で学習済み。エンコーダ変更後は次元が変わるため(776→904)、これらのチェックポイントは新コードでは`--resume`もtools/run_battle.pyでの再生もできない(`RuntimeError: size mismatch`になる)。新エンコーダで学習を再開する場合は新規run(エピソード0から)になる。
- 次にやること: 新エンコーダ確定後、新規runとして学習を再開する。対戦相手の重み(selfplay=0.35, crustle=0.25, lucario_v1/v2/iono/abomasnow=各0.1, random=0)は変更不要なはずだが、確定時に再確認する。

### 2026-06-22 17:52 — `encoder_v2_20000ep_20260622_1752`(本番実行・進行中)

上記4特徴量(ATTACKの弱点/抵抗補正後実効ダメージ+KO可能フラグ、相手ベンチのスカラー特徴、スタジアムカード特徴、相手の捨て札プール埋め込み)をCodex CLI(`agent/rl_agent.py`を編集)で追加し、自分で独立に動作検証(`py_compile`、ランダムプレイのスモークテスト、Magcargo ex(炎)→Pinsir(弱点炎)での弱点×2補正のユニットテスト、`train_ppo.py --episodes 2`の完走)を済ませた上で本番runとして起動。

- エンコーダの`concat_dim`は904→**1417**(`EMBED_DIM*3 + HIDDEN_DIM*4 + 9`)に変化。**既存チェックポイント(776次元・904次元のどちらも)は全て新コードと次元が合わず`--resume`不可**。このrunはエピソード0からの新規学習。
- コマンド: `uv run python -u train_ppo.py --episodes 20000 --save_dir models/encoder_v2_20000ep_20260622_1752`(他は全てデフォルト値。デフォルトの対戦相手重みが既に`rebalanced`系の検証済み設定: selfplay=0.6, crustle=0.15, lucario_v1/v2/iono/abomasnow=各0.05, random=0.05)
- 起動方法: `nohup ... & disown`
- ログ: `agent/logs/encoder_v2_20000ep_20260622_1752.log`(ローカルのみ、gitignore対象)
- チェックポイント: `agent/models/encoder_v2_20000ep_20260622_1752/`にepisode 500刻みで保存予定
- 状態: 起動直後(EP0)。クラッシュなく開始したことのみ確認済み。
- 目的: 特徴量追加(レバー1: 入力表現の強化)が、レバー2(ヒューリスティック/RL分担によるアクション空間分解)に着手する前の時点でどこまで学習効率を改善するかを単独で確認する。次にやること: 一定エピソード進んだ時点で対戦相手別勝率を確認し、旧776次元runの傾向(ヒューリスティック相手に全敗)から改善しているか比較する。

修正後、`model_ep5000.pt`から`--resume`で再開済み(2026-06-20 17:23〜)。

- 状態: `rebalanced_20000ep_20260619_1849`は進行中(EP5000から再開)。完了したら20戦評価を追記する。

### 2026-06-22 20:38 — `encoder_v2_20000ep_20260622_1752`を停止 → `encoder_v3_20000ep_20260622_2038`へ切替

`encoder_v2_20000ep_20260622_1752`実行中に、ネットの定石(JustInBasil/PokeBeach/TCG Protectors等)を参照しながら`agent/rl_agent.py`へ特徴量を3ラウンド追加(いずれもCodex CLIで実装し自分で独立検証済み)。Pythonのimport仕様上、実行中プロセスはディスク変更を読み直さないため、これらの変更は`encoder_v2_20000ep_20260622_1752`には反映されていなかった。ユーザー判断で同runを停止し、全特徴量を含む新規runに切り替える。

**追加した特徴量(ラウンド2〜4。ラウンド1は`encoder_v2_20000ep_20260622_1752`起動時点で既に入っている):**

- ラウンド2(`POKE_SCALAR_DIM` 23→24, `global_scalars` 9→14): 既存の正規化不整合を修正(prize/turn/bench/エネルギー数を他の特徴と同スケールに統一)。`_max_affordable_damage`(現在のエネルギーで実際に打てるワザの中で弱点/抵抗補正後の最大実効ダメージ)を追加し、自分/相手のアクティブ・ベンチ全体に対称適用(「今受けたら即死するか」「ベンチに置いたこの子はガストKOされる射程内か」の両方に使える)。hand count(自分/相手)、先攻フラグ、ready_attacker数(自分/相手)を追加。
- ラウンド3(`POKE_SCALAR_DIM` 24→28): ユーザー提案により、`CardData.evolvesFrom`の逆引きインデックスを構築し、種ポケモンが将来進化する可能性のある最大HP・最大ワザダメージ・ex化するかを特徴量化(`_load_evolution_index`/`_future_evolutions`/`_evolution_threat_feats`)。相手のデッキを知らなくてもカードDB自体の進化関係から導けるため汎用的。
- ラウンド4(`POKE_SCALAR_DIM` 28→29, `global_scalars` 14→19): スタジアムの所有者フラグ(`Card.playerIndex`から「自分が出したか」を判定。Stadium war定石対応)、このターンに使った行動4種(`supporterPlayed`/`energyAttached`/`retreated`/`stadiumPlayed`)、ツール装備フラグ(`has_tool`)を追加。

**エンコーダの次元変化:** `concat_dim` 1417→**1427**、`POKE_SCALAR_DIM` 23→**29**。既存チェックポイント(776次元・904次元・1417次元のいずれも)は全て新コードと次元が合わず`--resume`不可。

- 停止した`encoder_v2_20000ep_20260622_1752`の最終状態: EP2420まで到達。EP2400時点の評価(10戦ずつ): `random=60.0%, lucario_v1=0.0%, lucario_v2=20.0%, crustle=0.0%, iono=0.0%, abomasnow=10.0%`。ラウンド1の特徴量(弱点/抵抗ダメージ・相手ベンチ特徴・スタジアム・相手捨て札)だけでは、この時点でもヒューリスティック相手にはほぼ勝てていない。
- 新run: `uv run python -u train_ppo.py --episodes 20000 --save_dir models/encoder_v3_20000ep_20260622_2038`(設定は前回と同じデフォルト値)。起動直後(EP0)でクラッシュなしのみ確認済み。
- ログ: `agent/logs/encoder_v3_20000ep_20260622_2038.log`(ローカルのみ)
- 次にやること: 一定エピソード進んだ時点で評価し、`encoder_v2`のEP2400時点の数字と比較してラウンド2〜4が効いているか判断する。

**2026-06-23 進捗確認(EP10000/20000、半分到達)**: `eval winrate(10 games each)`: `random=80.0%, lucario_v1=0.0%, lucario_v2=10.0%, crustle=0.0%, iono=0.0%, abomasnow=0.0%`。`encoder_v2`のEP2400時点(`random=60.0%, lucario_v1=0.0%, lucario_v2=20.0%, crustle=0.0%, iono=0.0%, abomasnow=10.0%`)と比べて、4倍以上のエピソードを使っても明確な改善は見えていない(むしろlucario_v2/abomasnowは下がっている)。**ラウンド2〜4の特徴量追加だけでは、ヒューリスティック相手への勝率という意味では今のところ目立った効果が出ていない**ことを正直に記録しておく。crustle(ウォール)に対しては相変わらず0%で、これは「弱点/抵抗の脅威評価」を入れても、Crustleのような特性無効化型のウォールには別の対策(特性ロック耐性や長期戦プラン)が必要であることを示唆している可能性がある。クラッシュは無く、20000エピソード完走まで継続する。

**2026-06-23 完了(EP20000、`model_final.pt`保存済み)**: 終盤(EP19200〜19800)の評価を並べると `random` は `40%→80%→40%→30%` と大きく振れている(評価が10戦ずつなのでサンプルが少なく分散が大きいことに注意。「下がった」と断定はできない)。一方で**ぶれていない事実**として、EP9800以降(エピソード半分以降)、`crustle=0.0%`が文字通り全評価ポイントで継続し、`lucario_v1`も常に`0.0%`、`lucario_v2`/`abomasnow`/`iono`も大半が`0%`(まれに数%)で、**学習後半1万エピソード分でヒューリスティック相手への勝率に意味のある改善が見られなかった**。

これは前述のCrustleのex無効化特性の発見(`agent/dragapult_agent_v2.py`のFEATURE_SPECS.mdラウンド8〜9参照)と整合する: RL側の特徴量(ラウンド1〜4)は弱点/抵抗ベースの実効ダメージ計算はカバーしているが、「特性によるダメージ無効化」という概念を表現する特徴量を持っていないため、`encoder_v3`もCrustleに対して有効な手筋を学習する手がかりが無かったと考えられる。次にやること: RL側にも`agent/dragapult_agent_v2.py`の`EX_DAMAGE_IMMUNE_IDS`と同様の「相手アクティブ/ベンチがex無効化特性を持つか」の特徴量を追加した上で、改めて学習を再開するのが筋が良さそうだが、ユーザー判断待ち。

### 2026-06-23 `agent/train_bc.py`によるBehavior Cloning事前学習

ユーザーから「自分の行動を分岐させて勝利になる行動を学習できないか」という相談を受け、
3方向(MCTSをより本格化/探索結果を学習データにして方策を改善/その他)を提示し、
「探索結果を学習データにして方策自体を改善する(AlphaZero的)」を選択。完全なMCTS+NN
co-trainingは大規模すぎるため、現実的な第一歩として、**search_begin統合済みの
`dragapult_agent_v2`の意思決定をBehavior Cloningの教師データとして`agent/rl_agent.py`の
`PTCGNet`を事前学習する**スクリプト`agent/train_bc.py`を新規作成した。

- データ収集: `dragapult_agent_v2`同士の自己対戦+`train_ppo.py`の`FIXED_OPPONENTS`
  (random/lucario_v1/v2/crustle/iono/abomasnow)との対戦から`(obs_dict, 選んだaction)`を収集。
- 学習: `_sequential_log_prob`(既存のPPO実装と同じ、複数選択を逐次マスクして扱う仕組み)で
  教師の選択に対する負対数尤度を損失として`PTCGNet`を更新。
- 出力チェックポイントは`PPOTrainer.save/load`と互換の形式({"model","optimizer","episode"})。
  `PPOTrainer.load()`で実際に読み込めることを確認済み。

**初回実行**(`models/bc_pretrain_v1.pt`、自己対戦150戦+対戦相手プール各15戦+3epoch、
合計32189サンプル): `avg_loss`は`1.0041→0.9997→0.9986`とほぼ変化しなかったが、これは
「選択肢が1つしかない自明な場面(損失0、勾度0)」が全体の約15%(200ステップ中24)を占めて
平均を薄めていたためと判明。**選択肢が複数ある場面(本当に学習が必要な場面)だけで
測ると、学習後のネットワークの最有力候補(top1)がヒューリスティック(search込み)の
実際の選択と77%(140件中108件)一致**しており、ランダムな初期状態よりはるかに高い
一致率を達成していることを確認した。

次にやること: この`models/bc_pretrain_v1.pt`を`train_ppo.py --resume`の初期値として
PPO自己対戦を開始する(ユーザー判断待ち)。チェックポイントファイル自体は
`agent/models/`配下なのでgitignore対象、スクリプトのみコミットする。

### 2026-06-25 重大バグ発見・修正: PTCGNetのvalue出力が学習中に発散していた

`agent/mcts.py`/`agent/train_mcts.py`(MCTS+自己対戦学習、別エントリ参照)の検証中、
`models/bc_pretrain_v1.pt`を初期値にした学習でvalue損失がepoch0で**150億超**という
異常値になった。直接デバッグした結果、`bc_pretrain_v1.pt`自体が既に壊れていたことが
判明した:

```
新規ランダム初期化のPTCGNet:    value=0.06(正常)
bc_pretrain_v1.ptを読み込み後:  value=2,130,240、state_vec abs mean=10,193,817(異常)
```

原因は2つの複合: (1) `agent/rl_agent.py`の`PTCGNet.value_head`の出力に上限を設ける
活性化(tanh等)が無かった、(2) `agent/train_bc.py`の学習ループに勾度クリッピングが
無く、サンプル1個ずつ・学習率1e-3のSGDを32189サンプル×3epoch回す過程で重みが発散した。
`avg_loss`の表示自体は1.0前後で「正常そう」に見えていたが、これは交差エントロピー
損失(softmaxは全体に定数を足しても不変なので、logitsの絶対値が大きくても相対的な
大小関係が保たれていれば損失は小さく見える)が発散を隠してしまっていたため。
behavior cloningの一致率(77%)が良好だったのも同じ理由で、policy側の相対的な順序は
保たれていたためたまたま実害が小さかった。**value側は相対値で正規化されないMSE損失を
直接使うため、この発散がそのまま致命的な損失爆発として表面化した**。

**修正**(Codex CLIで実装、独立検証済み):
1. `agent/rl_agent.py`: `value = torch.tanh(self.value_head(state_vec))` — value出力を
   [-1,1]に制限(標準的なAlphaZero系実装の流儀に合わせた)。
2. `agent/train_bc.py`: `loss.backward()`後、`optimizer.step()`前に
   `torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)`を追加。
3. `agent/train_mcts.py`: 同様の勾度クリッピング、`--value_coef`(デフォルト0.5、
   policy損失とのスケールを揃える)、`--accum_steps`(デフォルト32、勾度累積による
   実質的なミニバッチ化)を追加。`--lr`のデフォルトも1e-3→3e-4に変更。

検証: 修正後に`train_bc.py --games 5 --epochs 2`を再実行し、学習後のチェックポイントで
`state_vec`のabs meanが`0.023`(修正前は`10,193,817`)、valueが`-0.075`(`[-1,1]`の範囲内)
であることを確認した。

**影響範囲の補足**: この発散は`agent/train_bc.py`が生成する`bc_pretrain_v1.pt`に
限定された問題で、`agent/train_ppo.py`(`encoder_v2`/`encoder_v3`等)はBCを経由しない
ゼロからの学習だったため、この特定のバグの影響は受けていない。ただし
`PTCGNet.value_head`にtanhを追加したことで、今後`train_ppo.py`でPPOを学習する際にも
value推定が[-1,1]に制限されるようになる(GAE計算等への影響は今後の学習runで観察する)。

次にやること: `bc_pretrain_v1.pt`を修正後のコードで作り直し(`bc_pretrain_v2.pt`)、
それを初期値にしてMCTS自己対戦学習(`agent/train_mcts.py`)を再実行する。

### 2026-06-26 value発散バグの追加修正2件、ようやく健全な学習を確認

`bc_pretrain_v2.pt`から再開したMCTS学習で、`avg_value_loss`がepoch間で完全に同一値
(`1.9571`)に固まる現象が発生。チェックポイントのvalue出力を200局面で調べたところ
**全て-1.0固定**(degenerate)だった。原因はtanh飽和による勾度消失(`value_head`の
最終層が大きい値を出し、tanh(-大きな値)≈-1付近で勾度がほぼ0になっていた)。

**修正1**: `agent/rl_agent.py`の`value_head`最終層を小さい重み(`uniform(-0.01,0.01)`)・
ゼロバイアスで初期化。`bc_pretrain_v3.pt`を作り直したが、**まだdegenerate(-1.0固定)**
だった。

調査すると、value_head自身の重みは小さいまま(`weight abs max=0.00999`)だったが、
**入力側のstate_vec(StateEncoderの出力)のabs meanが36182**まで肥大化していた。
勾度クリッピング(max_norm=1.0)は既に入っていたが、1ステップごとのノルム制限だけでは
9万回超(150試合×3epoch、サンプル1個ずつの更新)の累積的なドリフトを防げなかった。
ネットワーク全体に正規化層(LayerNorm等)が一切無かったことが根本原因。

**修正2**: `agent/rl_agent.py`の`StateEncoder.forward()`の出力に`nn.LayerNorm(STATE_DIM)`
を追加。これにより新しい学習可能パラメータが増えるため、既存チェックポイントは
読み込み非互換になった(想定済み)。

`bc_pretrain_v4.pt`を作り直して検証: 150局面中48種類の異なるvalue出力(範囲0.02〜0.05、
degenerateでない)を確認。続けて`mcts_gen1_v4_20260626_0241`(150試合・24ワーカー・
32シミュレーション・3epoch)を実行した結果:

```
epoch=0 avg_loss=1.8971 avg_policy_loss=1.4466 avg_value_loss=0.9010
epoch=1 avg_loss=1.6271 avg_policy_loss=1.2879 avg_value_loss=0.6783
epoch=2 avg_loss=1.7264 avg_policy_loss=1.5131 avg_value_loss=0.4266
```

`avg_value_loss`が単調に改善(0.90→0.68→0.43)、degenerateな崩壊なし。**数値的に
健全なMCTS自己対戦学習が初めて成立した**。

ただし`train_ppo.py`の`evaluate()`(MCTSを使わず生の方策のargmaxで評価)で対戦相手
プールへの勝率を見ると、`random=45.0%`、`lucario_v1=0.0%`、`lucario_v2=5.0%`、
`crustle=0.0%`、`iono=0.0%`、`abomasnow=0.0%`と、まだ強くはない。これは1世代
(150試合)分の学習データでは想定通り(AlphaZero方式は世代を繰り返すことで強くなる
設計のため)。今回の主成果は「数値的に安定して学習が回る基盤が整ったこと」であり、
次にやることは複数世代の反復(このネットでまた自己対戦データを集めて再学習、を
繰り返す)。

### 2026-06-26 GPUバッチ化+世代ループを実装 → policy側でも同種の活性化発散が再発

ユーザー希望(「CPU・GPUを最大半分程度、6〜12時間程度の学習」)に基づき、Codex CLIで
以下を実装: (1) `agent/rl_agent.py`の`PTCGNet.forward`に`action_mask`引数を追加し
バッチ内で選択肢数が異なるサンプルを同時に扱えるようにする、(2) `agent/train_mcts.py`の
`train()`をサンプル1個ずつの逐次更新からミニバッチ化(`--batch_size`、デフォルト64)、
(3) `--generations`オプションで「自己対戦データ収集→学習」を指定回数自動で繰り返す
ループを追加。小規模テスト(8試合・2epoch)では`avg_value_loss`が健全に推移し問題なし
と判断、本番ジョブ(`--games 300 --workers 16 --simulations 64 --epochs 4
--batch_size 128 --generations 14`、`mcts_gen1_v4`から再開)をバックグラウンドで起動した。

14世代完走後にログを確認すると、**後半世代で`avg_value_loss`がepoch間で完全に同一値
(`2.0966`等)に固まり、valueが1.0に飽和**(前回と同じdegenerateパターン)していた。
チェックポイントを直接調べると以下が判明:

```
bc_pretrain_v4.pt:                logit絶対値最大 ≈ 590万
mcts_gen1_v4(今回の再開元):        logit絶対値最大 ≈ 1.9億
mcts_loop_gen6.pt(今回学習中):     logit絶対値最大 ≈ 115億
mcts_loop_gen9.pt:                 logit絶対値最大 ≈ 600億
mcts_loop_gen10〜13.pt:            logit絶対値最大 ≈ 970億、value=1.0固定(完全に劣化)
```

**根本原因**: 前回(2026-06-26の1つ前のエントリ)の修正で`StateEncoder`の出力(value側
の入力)にはLayerNormを追加したが、`PTCGNet.forward`内の`action_proj`
(行動特徴量→STATE_DIM)から`policy_attn`→`policy_head`に至るpolicy側の経路には
正規化が一切無かった。つまり**value側の発散を塞いだだけで、policy側に全く同じ脆弱性
(正規化層が無い経路は勾度クリッピングだけでは長時間学習の累積ドリフトを防げない)が
残っていた**。さらに調査すると、この発散は今回の14世代学習で新たに始まったのではなく、
`bc_pretrain_v4.pt`の時点(初回のBC学習直後)から既にlogitが590万という異常スケールで
存在しており、これまでのラウンドでは(softmaxは相対値のみに依存するため)policyの
振る舞い自体には大きな実害が出ておらず気づかれていなかった。学習を重ねるほど指数的に
スケールが増大し、今回ついしてvalue側のtanh飽和という形で表面化した。

**修正**(Codex CLIで実装): `PTCGNet.__init__`に`self.action_norm = nn.LayerNorm(STATE_DIM)`
(action_proj直後)と`self.policy_norm = nn.LayerNorm(STATE_DIM)`(`policy_attn`の出力と
`action_proj`の合計に対して、`policy_head`の直前)を追加。action_normだけでは
`policy_attn`が再び大きく増幅してしまうため(検証時に約59万まで増幅)、2箇所目の
policy_normが必要だった。

**追加で判明した重要な事実**: 修正後、既存の発散済みチェックポイント(`mcts_gen1_v4`等)
を`strict=False`で読み込んで追加学習を試したところ、`avg_policy_loss`がepochを通じて
**完全に同一値**(例: `1.4603`が3epoch連続)になる、つまり一切学習が進まない現象が発生
した。一方、同じ条件でランダム初期化(チェックポイント無し)から学習すると、
`avg_policy_loss`はわずかながら実際に減少した(`1.6562→1.6541→1.6532`)。
**LayerNormの勾度はそのままだと入力の分散に反比例して縮小する(`grad ∝ 1/σ`)ため、
既に大きく発散した重みの上にLayerNormを後付けすると、forward時の出力スケールは
正常化されてもbackward時の勾度がほぼゼロになり、学習が実質的に凍結してしまう**。
つまりLayerNorm追加は「新規にゼロから学習する場合の発散防止」には効くが、「既に発散した
重みを後から正常化する」ことはできない。

**対応**: 発散済みの旧チェックポイント系列(`bc_pretrain_v1〜v4`、`mcts_gen1_v4`、
`mcts_loop*`)は破棄し、修正後アーキテクチャで`bc_pretrain_v5.pt`をゼロから作り直した
(`train_bc.py --games 200 --vs_opponent_games 20 --epochs 3`)。検証: value範囲
`[0.02, 0.09]`(多様・有界)、logit絶対値最大`15〜162`(常識的な範囲)。これを起点に
本番のMCTS世代学習(`models/mcts_loop_v2.pt`、同じパラメータで`--generations 14`)を
再起動し、gen0で`avg_policy_loss`が`1.55→1.38→1.37→1.37`と実際に減少していることを
確認済み(凍結していない)。

**教訓**: 正規化層を新規に追加する修正は、既存の発散済みチェックポイントへの
`--resume`では効果が無い(LayerNormのbackward勾度縮小により学習が凍結する)ことがある。
正規化層の追加・変更を行った際は、`--resume`した直後の数epochで損失が実際に動いている
ことを毎回確認する必要がある(凍結に気づかず長時間学習を回すと、今回のように発散が
再発していることに気づくのにさらに時間がかかる)。
繰り返す)。
