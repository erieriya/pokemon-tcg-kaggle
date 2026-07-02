# セッション引き継ぎノート

> **このファイルの目的**: 新しいClaude/作業セッションが迷わず続きから作業できるようにする。
> コードを読めば分かることより「なぜそうなっているか」「何に気をつけるか」を重視する。
> 最終更新: 2026-07-01

---

## 1. プロジェクト概要

**何をしているか**: Kaggle「PTCG AI Battle Challenge Simulation」(`pokemon-tcg-ai-battle`)向けのAIエージェントを開発中。cg(対戦シミュレータ)上でDragapult ex/Dusknoir構成デッキを動かす。

**現在のスコア(リーダーボード)**: 3604位/3740チーム、スコア285.1(下位3.6%)。上位の中央値は674.8。詳細は`KAGGLE_RESEARCH.md`参照。

**リポジトリ**: `github.com:erieriya/pokemon-tcg-kaggle.git`

---

## 2. ファイルマップ

### 提出関連
| ファイル | 役割 |
|---|---|
| `submit.sh "コメント" [agent/xxx.py]` | Kaggle提出スクリプト。第2引数省略時はdragapult_agent.pyを使う |
| `agent/dragapult_agent_v2.py` | **現在のメインのヒューリスティックエージェント**。これを`submit.sh`で提出する |
| `agent/dragapult_agent.py` | オリジナル版(v2の前身)。submit.shのデフォルト |
| `agent/deck.csv` | 提出用デッキ(Dragapult ex/Dusknoir) |

### 学習パイプライン
| ファイル | 役割 |
|---|---|
| `agent/rl_agent.py` | PTCGNetのアーキテクチャ定義 + `encode_state`/`encode_actions` |
| `agent/train_bc.py` | Behavior Cloning事前学習(dragapult_agent_v2を教師に) |
| `agent/train_mcts.py` | MCTS自己対戦+学習ループ(`--vs_opponents`で固定相手との対戦も可) |
| `agent/mcts.py` | PUCTツリーMCTS実装(`mcts.search_policy`が公開API) |
| `agent/eval_mcts.py` | 学習済みチェックポイントをFIXED_OPPONENTSと対戦評価 |

### 対戦相手プール(train_ppo.py の FIXED_OPPONENTS)
現在8体: `random`, `lucario_v1`, `lucario_v2`, `crustle`, `iono`, `abomasnow`, `alakazam`, `archaludon`
- `alakazam_agent.py` / `archaludon_agent.py`: 2026-06-30に追加(Kaggle公開notebook移植)
- 対応デッキCSV: `deck_alakazam.csv`, `deck_archaludon.csv` 等

### ドキュメント
| ファイル | 内容 |
|---|---|
| `FEATURE_SPECS.md` | ラウンドごとの実装履歴(ラウンド1〜17+) |
| `TRAINING_LOG.md` | 学習実行ログ(バグ記録含む) |
| `KAGGLE_RESEARCH.md` | リーダーボード状況・メタ・ロードマップ |
| `SESSION_NOTES.md` | このファイル。セッション引き継ぎ用 |

---

## 3. 現在の状況(2026-07-02時点)

### アーキテクチャ現状
- **POKE_SCALAR_DIM: 49** (旧32から拡張。2026-07-01変更)
  - 追加: card_energy_type 1-hot(12), weakness(1), resistance(1), can_attack_now(1), max_outgoing_damage(1), appear_this_turn(1)
- **global_scalars: 24次元** (旧19から拡張)
  - 追加: can_ko_opp(1), opp_can_ko_me(1), prize_diff(1), turnActionCount(1), opp_discard_ex_count(1)
- **⚠️ bc_pretrain_v6以前のチェックポイントは旧アーキと非互換** (形状不一致でload不可)

### 進行中の学習
```
bc_pretrain_v8: PID 2711288 (学習中 - epoch 0完了、epoch 1〜2残り)
コマンド: train_bc.py --games 200 --vs_opponent_games 20 --vs_hard_opponents crustle,abomasnow --vs_hard_games 100 --epochs 3 --out models/bc_pretrain_v8.pt
ログ: /tmp/bc_pretrain_v8_run.log
注意: この教師データはv2改善前に収集済み(旧dragapult_agent_v2)

bc_pretrain_v9: ウォッチャー起動中(v8完了後に自動起動)
コマンド: 同じ引数で --out models/bc_pretrain_v9.pt
ログ(予定): /tmp/bc_pretrain_v9_run.log
注意: v9は改善後のdragapult_agent_v2(2026-07-02 commit abf0970)を教師に使用

mcts_loop_v9: ウォッチャー起動中(bc_v9完了後に自動起動)
コマンド: train_mcts.py --resume models/bc_pretrain_v9.pt --out models/mcts_loop_v9.pt
  --simulations 200 --temperature 1.0 --temperature_end 0.3
  --generations 15 --vs_opponent_games 75 --workers 16
ログ(予定): /tmp/mcts_loop_v9_run.log
```

### dragapult_agent_v2 改善内容(2026-07-02)
Crustle(EX耐性)対策を強化(commit abf0970):
- OPT_ATTACK(EXが耐性持ちを攻撃): score -50 → **-500** に変更
- エネルギーアタッチ+120ボーナス: `crustle_on_bench` → `crustle_ex_immune_present`(=activeでも判定)
- Crustle active + 自分EXのリトリート: +80ボーナス追加
- SWITCH/TO_ACTIVE(ctx=3,4): Crustle active時はDusknoir優先 → Dusknoir→Dusclops→Dreepy順
- Dusknoir/Dusclops ability: blocked_wall時のボーナス 50/30 → 100/60 に増加

### mcts_loop_v8 最終結果
```
最終世代(gen9, temperature=0.300):
  自己対戦: wins=148 losses=152 (50/50)
  vs crustle: wins=0 losses=75 (Crustle問題未解決)
  vs abomasnow: wins=2 losses=73
  vs alakazam: wins=16 losses=59 (27%)
  vs archaludon: wins=3 losses=72
  policy_loss: 1.4876 (uniform ≈ 4.4候補)
  value_loss: 0.074
```

### Kaggle提出状況
- 2026-07-01 03:48頃提出: "PTCGNet+PUCT MCTS (mcts_loop_v5 gen13, 64sims, ability features POKE_SCALAR_DIM=32)"
  - `submit_mcts.sh` で mcts_loop_v5.pt を提出。スコア確定待ち。
- 確認: `kaggle competitions submissions pokemon-tcg-ai-battle`

### チェックポイント一覧
| チェックポイント | 状態 | 備考 |
|---|---|---|
| `models/bc_pretrain_v7.pt` | ✅ 完了 | POKE_SCALAR_DIM=49新アーキ。2026-07-01 04:17完了 |
| `models/bc_pretrain_v8.pt` | 🔄 学習中 | 旧v2教師。epoch 0 avg_loss=1.03 |
| `models/bc_pretrain_v9.pt` | ⏳ 待機 | 改善v2教師(abf0970)で自動起動 |
| `models/mcts_loop_v7.pt` | ✅ 完了(12世代) | bc_v7ベース。value_loss 0.10まで改善 |
| `models/mcts_loop_v8.pt` | ✅ 完了(10世代) | v7ベース、sim=128、温度annealing 1.0→0.3 |
| `models/mcts_loop_v9.pt` | ⏳ 待機 | bc_v9ベース、sim=200、rule-based mask有効 |
| `models/mcts_loop_v5.pt` | ✅ 完了(14世代) | 旧アーキ(POKE_SCALAR_DIM=32)。Kaggle提出済み |
| `models/bc_pretrain_v6.pt` | ⚠️ 旧アーキ(POKE_SCALAR_DIM=32) | 現在のコードと非互換 |
| `models/bc_pretrain_v5.pt`以前 | ❌ さらに旧アーキ | 使用不可 |

---

## 4. 重要な発見・落とし穴(知らないとはまる)

### 【最重要】Kaggle提出環境の exec() 問題
**症状**: 提出後すぐ`status: ERROR`、スコアが付かない。
**原因**: Kaggleはmain.pyを`exec(code_object, env)`で実行するため`__file__`が未定義。
`os.path.abspath(__file__)`を書くと`NameError: name '__file__' is not defined`で即クラッシュ。

**正しい書き方**:
```python
try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = "/kaggle_simulations/agent"  # Kaggle exec()環境でのフォールバック
sys.path.insert(0, _AGENT_DIR)
```
→ `dragapult_agent.py`/`dragapult_agent_v2.py`には既に適用済み。
→ `rl_agent.py`はモジュールとしてimportされる前提なので`__file__`は使える(問題なし)。

**確認方法**: エピソードログの`statuses`が`["ERROR","ERROR"]`になっていないか。
ユーザーがKaggle提出詳細ページからエピソードJSONを確認できる。

### アーキテクチャ変更時は必ずBC事前学習から再スタート
`POKE_SCALAR_DIM`/`concat_dim`を変えると`nn.Linear`の重み形状が変わり、
`load_state_dict(strict=False)`でも解決しない(missing keyではなくshape mismatch)。
これまでの変更歴:
- `POKE_SCALAR_DIM`: 23→24→28→29→32→**49(現在)**
- `StateEncoder.concat_dim`: 内部で自動計算されるが、`PokemonEncoder.__init__`の
  `nn.Linear(embed_dim + POKE_SCALAR_DIM, ...)`が変わるため毎回再学習必要

### 活性化発散バグ(過去3回発生、全て修正済み)
1. **BC訓練の勾度クリッピング欠如**: `clip_grad_norm_(max_norm=1.0)`を追加済み
2. **StateEncoder出力にLayerNorm不足**: `output_norm = nn.LayerNorm(STATE_DIM)`追加済み
3. **action_proj/policy_headにLayerNorm不足**: `action_norm`/`policy_norm`追加済み
   
**症状**: `avg_value_loss`がepoch間で完全に同一値(e.g. 2.0966)に固定される。
**確認**: 評価時に`value`が常に1.0や-1.0に固定されていないか確認。

### 発散済みチェックポイントから再開するとLayerNorm勾度が消える
LayerNormを後付けしても、入力分散が大きすぎるとbackward勾度が`1/σ`スケールで縮小し、
学習が凍結する(policy_lossがepoch間で完全に同一値になる)。新アーキの変更後は
必ず**ゼロ(ランダム)から**BC再学習すること。既存の大きく発散した重みを引き継がない。

### multiprocessing(spawn方式)実行中はコードファイルを編集しない
`train_mcts.py`はworkerプロセスをspawnで起動し、各workerが**ディスクから`train_mcts.py`を
再importする**。実行中に`train_mcts.py`や`rl_agent.py`を編集すると、
次のworker起動時にmain/worker間でコードバージョンが不一致になりクラッシュする。
(実際に複数回発生: mcts_loop_v2が11世代目で中断、mcts_loop_v6がgen0でクラッシュ)。
**2026-07-01**: rl_agent.py(POKE_SCALAR_DIM変更)をv6学習中に編集→v6がgen0でクラッシュして0チェックポイント。

### `torch.set_num_threads(1)`は必ず入れる
複数workerが全コア分のスレッドをそれぞれ要求すると実測で48倍のパフォーマンス低下。
`play_one_game`/`play_vs_opponent_game`の冒頭に既に記述済み。

---

## 5. よく使うコマンド

### 学習
```bash
cd agent

# BC事前学習(新アーキ時の再スタート、約20分)
uv run python -u train_bc.py --games 200 --vs_opponent_games 20 --epochs 3 --out models/bc_pretrain_vX.pt

# MCTS自己対戦学習(本番設定)
nohup uv run python -u train_mcts.py \
  --games 300 --workers 16 --simulations 64 --candidates 4 --min_candidates 1 --dynamic_candidates \
  --max_steps 300 --epochs 4 --batch_size 128 --generations 14 --temperature 1.0 \
  --vs_opponents crustle,abomasnow,alakazam,archaludon --vs_opponent_games 75 \
  --resume models/bc_pretrain_vX.pt --out models/mcts_loop_vX.pt \
  > /tmp/mcts_loop_vX_run.log 2>&1 &

# 学習状況確認
tail -30 /tmp/mcts_loop_vX_run.log
```

### 評価
```bash
cd agent

# MCTS推論込みでFIXED_OPPONENTSと対戦評価
uv run python -u eval_mcts.py \
  --checkpoint models/mcts_loop_v5.pt \
  --games 15 --simulations 64 --candidates 4 --max_steps 300

# 複数チェックポイントまとめて比較
uv run python -u eval_mcts.py \
  --checkpoint models/mcts_loop_v5.pt models/mcts_loop_v4.pt \
  --games 15 --simulations 64
```

### 提出
```bash
# ヒューリスティックエージェント提出
./submit.sh "コメント" agent/dragapult_agent_v2.py

# 提出状況確認
kaggle competitions submissions pokemon-tcg-ai-battle

# リーダーボード確認(上位20)
kaggle competitions leaderboard pokemon-tcg-ai-battle -s --show
```

### チェックポイント健全性チェック
```bash
cd agent && uv run python - <<'EOF'
import torch, sys, os
sys.path.insert(0, "../data/sample_submission")
from cg.game import battle_start, battle_select, battle_finish
from rl_agent import PTCGNet, encode_state, encode_actions
from train_ppo import read_deck

ckpt = torch.load("models/mcts_loop_v5.pt", map_location="cpu")
net = PTCGNet()
net.load_state_dict(ckpt["model"], strict=False)
net.eval()
deck = read_deck("deck.csv")
obs_dict, _ = battle_start(deck, deck)
vals = []
for _ in range(30):
    state = obs_dict.get("current") or {}
    if state.get("result", -1) != -1:
        break
    sel = obs_dict.get("select")
    if sel is None:
        break
    options = sel.get("option") or []
    if not options:
        action = []
    else:
        st = encode_state(obs_dict, "cpu")
        af = encode_actions(options, obs_dict, "cpu")
        with torch.no_grad():
            logits, value = net(st, af)
        vals.append(value.item())
        action = [int(logits[0].argmax())]
    try:
        obs_dict = battle_select(action)
    except IndexError:
        break
battle_finish()
if vals:
    print("value range:", min(vals), "~", max(vals), "/ distinct:", len(set(round(v,4) for v in vals)))
    print("⚠️ degenerate!" if len(set(round(v,4) for v in vals)) <= 1 else "✅ healthy")
EOF
```

---

## 6. 次にやること(優先順)

1. **学習完了の確認(2026-07-01夜〜朝)**:
   - `mcts_loop_v6` (PID 2444261): `tail -20 /tmp/mcts_loop_v6_run.log` - 完了後eval_mcts.pyで評価
   - `bc_pretrain_v7` (PID 2445401): `tail -5 /tmp/bc_pretrain_v7_run.log` - 約21分で完了予定
   - `mcts_loop_v7` (PID 2447712): bc_v7完了後に04:17頃に手動起動済み
     ログ: `/tmp/mcts_loop_v7_run.log` (12世代、完了予定 ~09:45 JST)

2. **Kaggle提出スコア確認**: `kaggle competitions submissions pokemon-tcg-ai-battle`
   - 最新提出: "PTCGNet+PUCT MCTS (mcts_loop_v5 gen13)" (2026-07-01 03:50頃)
   - 直前提出: "heuristic v2 + sys.path fix" (2026-06-29) の結果も確認

3. **mcts_loop_v6/v7完了後にKaggle提出**: `bash submit_mcts.sh "..." agent/models/mcts_loop_v6.pt` or v7

4. **特徴量リサーチ**: `FEATURE_RESEARCH.md`参照(今セッションで作成予定)。
   次世代(v8)の特徴量候補をさらに網羅的に調査・整理する。

3. **MCTS-trained agentの提出**: `mcts_loop_v5.pt`が評価良好なら、それを
   `agent/mcts_agent.py`(既存stub)か新規ファイルとして整備してKaggleに提出。
   MCTSでの推論は1手あたり約0.8秒かかる(64シミュレーション時)が、
   Kaggle環境のactTimeout/runTimeout範囲内に収まるか確認が必要。

4. **メタに合った対戦相手追加**: Starmie・Hop/Trevenant(meta_snapshotで「必須ストレステスト」
   と指摘)をFIXED_OPPONENTSに追加検討。対応するKaggle公開ノートブックを`kaggle kernels pull`
   で取得して移植する(alakazam/archaludonと同様のフロー)。

5. **`challenge-strategy`コンペ調査**: `pokemon-tcg-ai-battle-challenge-strategy`
   (賞金$240k、締切2026-09-13)が別途存在。戦略分析ノートブックを提出する形式の模様。
   未エントリー。詳細は`KAGGLE_RESEARCH.md`参照。

---

## 7. アーキテクチャ現状

```
PTCGNet (rl_agent.py)
├── StateEncoder
│   ├── CardEmbedding (EMBED_DIM=128)
│   ├── HandEncoder (Attention Pooling)
│   ├── PokemonEncoder (embed_dim + POKE_SCALAR_DIM=49 → HIDDEN_DIM=256)
│   └── LayerNorm(STATE_DIM=256) ← 活性化発散対策
├── action_proj: EMBED_DIM → STATE_DIM
├── action_norm: LayerNorm(STATE_DIM) ← 活性化発散対策
├── policy_attn: MultiheadAttention
├── policy_norm: LayerNorm(STATE_DIM) ← 活性化発散対策
├── policy_head: Linear(STATE_DIM → 1)
└── value_head: Linear(STATE_DIM → HIDDEN_DIM → 1) + tanh

POKE_SCALAR_DIM = 49 (2026-07-01更新、旧32から+17)
  = hp_ratio+dmg(2) + エネルギー(12) + 状態異常(5) + 静的特徴(4) + 被ダメージ(1) + 進化脅威(4)
    + ツール(1) + 特性関連(3) + card_energy_type 1-hot(12) + weakness(1) + resistance(1)
    + can_attack_now(1) + max_outgoing_damage(1) + appear_this_turn(1)

global_scalars = 24 (旧19から+5)
  追加: can_ko_opp, opp_can_ko_me, prize_diff, turnActionCount, opp_discard_ex_count

重要: POKE_SCALAR_DIM変更により bc_pretrain_v6.pt以前 & mcts_loop_v5以前は全て非互換。
  現在の有効チェックポイント:
    bc_pretrain_v7.pt (POKE_SCALAR_DIM=49, 学習中)
    mcts_loop_v7.pt (bc_pretrain_v7から開始予定、学習中)
    mcts_loop_v6.pt (POKE_SCALAR_DIM=32のv5から継続、別アーキテクチャ)
```

**チェックポイント互換性マップ:**
| チェックポイント | POKE_SCALAR_DIM | 状態 |
|---|---|---|
| bc_pretrain_v7.pt | 49 | 🔄 学習中 |
| mcts_loop_v7.pt | 49 | ⏳ bc_v7完了後に開始 |
| mcts_loop_v6.pt | 32 | 🔄 学習中(v5から継続、候補2) |
| mcts_loop_v5.pt | 32 | ✅ 完了(14世代) → Kaggle提出済み |
| bc_pretrain_v6.pt | 32 | ✅ 完了 |
| mcts_loop_v4.pt以前 | 32 | ⚠️ 古いv5の前身 |
| bc_pretrain_v5.pt以前 | 29 | ❌ 旧アーキテクチャ |

---

## 8. 知っておくと便利なこと

- **Kaggleエピソードデータセット**: `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`に
  毎日の対戦ログが格納。manifest: `kaggle/pokemon-tcg-ai-battle-episodes-index`。
  実態は各1ファイル=1エピソードのJSON。TeamNamesで自チーム("Eriya")を検索可能だが、
  低ランクチームは出現頻度が低い(調査時に8日分で自チームのゲームが2件しか見つからなかった)。
  1日分のzipは約700MB(展開後2GB程度)で、当初21GBという情報は誤り。

- **スコアはElo/Glicko的な動的レーティング**: 提出直後から徐々に確定していく。
  `publicScore`はリーダーボードの"最良スコア"を表示するため、新しい(低い)スコアの
  提出後もリーダーボード上のスコアは改善されたもの(過去最高)が残ることがある。

- **cogのsearch APIの特性**: `search_step(search_id, action)`は同じsearch_idに対して
  何度呼んでも別の子ノードを生成可能(親が「消費」されない)。実機確認済み。
  ただし`search_end()`を呼ぶと**全searchIdが無効化**される(プロセス全体)。
  MCTSのツリー再利用はこの制約があるため、実際のゲーム進行と探索の仮想先読みが
  別世界のため実装困難(実際に試みたが、battle_select後のobs_dictとの不一致でIndexErrorが多発)。

- **Dragapult agentをimportする際**: `from cg.api import ...`がトップレベルにある。
  `train_bc.py`や学習スクリプトでimportする前に必ず`CG_PATH`(data/sample_submission)を
  `sys.path`に追加しておくこと。
