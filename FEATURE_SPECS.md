# 特徴量・ヒューリスティック仕様

> `agent/rl_agent.py`と`agent/dragapult_agent_v2.py`に対してCodex CLIに委任した、
> 特徴量およびヒューリスティック仕様の作業指示書を整理した参照ドキュメント。
> 各実装は完了・検証済みであり、今後同様の変更を行う際の参考、および仕様レビューに使用する。

## ラウンド1: RLエンコーダへの盤面・行動特徴量追加

### 対象ファイル

- `agent/rl_agent.py`
- 対象: `encode_state()`、`encode_actions()`、`StateEncoder`

`agent/encode_state.py`は未使用の類似実装、`submission/rl_agent.py`は
`submit_rl.sh`が生成するビルド成果物であるため、どちらも対象外とした。

### 解決したかった問題

- ATTACK候補が素のワザダメージしか持たず、弱点・抵抗適用後の実効ダメージや
  KO可能性を判断できなかった。
- 相手アクティブ以外の盤面情報が乏しく、相手ベンチの状態を方策へ入力できなかった。
- 場に出ているスタジアムを認識できなかった。
- 相手が捨て札で公開したカードを、デッキ傾向の推論材料として利用できなかった。

相手の捨て札は基本的に増加し続けるため、外部の履歴状態は持たず、
各時点の`obs_dict`に含まれる`discard`をそのまま使用する方針とした。

### 具体的な仕様

#### ATTACKの実効ダメージとKO可能フラグ

既存の行動特徴量はindex 0〜58を使用していたため、既存indexを変更せず、
次の2要素を末尾へ追加する。

```python
_OFF_ATK_EFF_DAMAGE = _OFF_ATTACH_REMAINING + 1  # 59
_OFF_ATK_WOULD_KO = _OFF_ATK_EFF_DAMAGE + 1     # 60
```

`encode_actions()`の選択肢ループ前に、次を一度だけ取得する。

- `my_index = current.yourIndex`
- 自分のアクティブの`CardData.energyType`
- 相手のアクティブの`CardData.weakness`、`resistance`、現在HP

ATTACKごとに`attack.damage`へ以下の補正を適用する。

```text
自分のタイプ == 相手の弱点: 実効ダメージ = damage * 2
自分のタイプ == 相手の抵抗: 実効ダメージ = max(0, damage - 30)
それ以外:                   実効ダメージ = damage
```

弱点・抵抗の比較は両方の値が`None`でない場合だけ行う。特徴量は次の通り。

```python
feats[0, i, _OFF_ATK_EFF_DAMAGE] = effective_damage / 300.0
feats[0, i, _OFF_ATK_WOULD_KO] = (
    1.0 if opp_hp > 0 and effective_damage >= opp_hp else 0.0
)
```

#### 相手ベンチのスカラー特徴

自分ベンチと同じ構造で、相手ベンチから以下を作成する。

- `opp_bench_ids`
- `opp_bench_scalar`
- `opp_bench_mask`

`StateEncoder`ではベンチの処理を次の再利用可能なメソッドへ切り出す。

```python
_pool_bench(self, ids, scalar, mask)
```

このメソッドはカード埋め込みとスカラーを`PokemonEncoder`へ通し、
mask付き平均を返す。自分と相手のベンチへ同じ処理を適用し、
`bench_vec`と`opp_bench_vec`を`combined`へ連結する。

#### スタジアムカード

`current.get("stadium", [])`から次を作成する。

- `stadium_id`: スタジアムがなければ0
- `stadium_present`: スタジアムがあれば1.0、なければ0.0

`stadium_present`は`global_scalars`末尾へ追加する。
`stadium_id`は状態dictへ次のshapeで追加する。

```python
torch.tensor([stadium_id], dtype=torch.long, device=device).unsqueeze(0)
```

`StateEncoder.forward()`ではカード埋め込みを取得し、`combined`へ連結する。

```python
stadium_vec = self.card_emb(state["stadium_id"]).squeeze(1)
```

#### 相手の捨て札プール埋め込み

60枚すべてが捨て札へ入る可能性を考慮し、上限を次の通りとする。

```python
OPP_DISCARD_CAP = 60
```

相手の`discard`から最大60件のcard IDを取り出し、0埋めした次のテンソルを
状態dictへ追加する。

- `opp_discard_ids`: `long`、shape `(1, 60)`
- `opp_discard_mask`: `float`、shape `(1, 60)`

`StateEncoder`へ単純な埋め込みのmask付き平均を行うメソッドを追加する。
捨て札はポケモン以外も含むため、`PokemonEncoder`は使用しない。

```python
_pool_embeddings(self, ids, mask)
```

生成した`opp_discard_vec`を`combined`へ連結する。

### 次元の変化

- `POKE_SCALAR_DIM`: 23のまま
- `global_scalars`: 8 → 9
- `concat_dim`: 904 → 1417
- 変更後:

```python
concat_dim = EMBED_DIM * 3 + HIDDEN_DIM * 4 + 9
```

内訳はhand・stadium・opponent discardの3埋め込み、自分/相手のactive・benchの
4ポケモン表現、global scalar 9要素。

## ラウンド2: 正規化統一と現在の攻撃脅威特徴

### 対象ファイル

- `agent/rl_agent.py`
- 対象: `encode_state()`、ネスト関数`get_pokemon_scalar()`、
  `global_scalars`、`POKE_SCALAR_DIM`、`StateEncoder.__init__`

### 解決したかった問題

- エネルギー数、サイド、ターン、ベンチ数で正規化スケールが不統一だった。
- 各ポケモンが相手の現在の攻撃で受ける最大実効ダメージを認識できなかった。
- 手札枚数、先攻プレイヤー、現在攻撃可能なポケモン数が未入力だった。

攻撃可能性は、現在付いているエネルギーだけで要求を満たすワザに限定する。
弱点・抵抗補正はラウンド1と同じく、弱点2倍、抵抗-30、下限0とする。

### 具体的な仕様

#### 既存値の正規化

`get_pokemon_scalar()`のエネルギータイプ別枚数を次のように正規化する。

```python
energy_vec = [v / 4.0 for v in _energy_count_vec(poke.get("energies"))]
```

`global_scalars`では次の除数を使用する。

```text
自分/相手のprize残り枚数: / 6.0
turn:                       / 50.0
自分/相手のbenchサイズ:    / 5.0
benchMax:                   / 5.0
deckCount:                  / 60.0（既存仕様を維持）
stadium_present:            0/1（変更なし）
```

#### 最大実効ダメージの汎用ヘルパー

モジュールレベルに次の関数を追加する。

```python
def _max_affordable_damage(
    card_db,
    attack_db,
    attacker,
    defender_weakness,
    defender_resistance,
):
    """現在のエネルギーで使用可能なワザの最大実効ダメージを返す。"""
```

仕様は次の通り。

- 攻撃側カードは`attacker.get("id") or attacker.get("cardId")`で解決する。
- ワザの要求エネルギーを`_remaining_energy_cost(current, None, required)`で判定する。
- 戻り値が0のワザだけを使用可能とする。
- 攻撃側の`CardData.energyType`と防御側の弱点・抵抗を比較して補正する。
- 使用可能なワザがなければ`0.0`を返す。

`encode_state()`冒頭で`card_db`と`attack_db`をロードし、
ネスト関数からクロージャ経由で参照する。

#### 各ポケモンへのincoming damage追加

`get_pokemon_scalar()`を次のシグネチャへ拡張する。

```python
get_pokemon_scalar(poke, status=None, threat_active=None)
```

`threat_active`から対象ポケモンへ与えられる最大実効ダメージを計算し、
戻り値末尾へ追加する。

```python
incoming_max_damage / 300.0
```

`threat_active is None`の場合は0.0とする。呼び出しの対応は次の通り。

| エンコード対象 | `threat_active` |
|---|---|
| 自分のアクティブ | 相手のアクティブ |
| 相手のアクティブ | 自分のアクティブ |
| 自分のベンチ各体 | 相手のアクティブ |
| 相手のベンチ各体 | 自分のアクティブ |

#### global scalarの追加

既存リスト末尾へ次の5要素を追加する。

```python
my_hand_count = my.get("handCount", 0) / 20.0
opp_hand_count = opp.get("handCount", 0) / 20.0
is_first_player = 1.0 if current.get("firstPlayer") == your_idx else 0.0
my_ready_attackers = my_ready_count / 6.0
opp_ready_attackers = opp_ready_count / 6.0
```

`ready_attacker`はactiveとbenchを対象に、
`_max_affordable_damage(..., weakness=None, resistance=None) > 0`となる
ポケモンの数とする。ここでは弱点・抵抗補正を行わず、
現在のエネルギーでワザを1つ以上使えるかだけを表す。

### 次元の変化

- `POKE_SCALAR_DIM`: 23 → 24
- `global_scalars`: 9 → 14
- `concat_dim`: 1417 → 1422
- 変更後:

```python
concat_dim = EMBED_DIM * 3 + HIDDEN_DIM * 4 + 14
```

## ラウンド3: 進化先予測特徴

### 対象ファイル

- `agent/rl_agent.py`
- 対象: `get_pokemon_scalar()`、`POKE_SCALAR_DIM`、
  カードDB関連のモジュールレベルヘルパー

### 解決したかった問題

場にいるBasicポケモンの現在値だけでは、将来Stage 1・Stage 2やexへ進化した際の
脅威を評価できなかった。相手デッキの内容が不明でも、
カードDBの`CardData.evolvesFrom`を逆引きすれば進化候補を推定できるため、
全ポケモンへ対称に適用する方針とした。

### 具体的な仕様

#### 進化逆引きインデックス

既存のカード・ワザDBと同じキャッシュ方式で追加する。

```python
_EVOLUTION_INDEX: dict | None = None

def _load_evolution_index() -> dict:
    """進化前のカード名から、そのカードから進化するCardData一覧を引く辞書。"""
```

全`CardData`を走査し、`evolvesFrom`があるカードを次の形で格納する。

```python
index.setdefault(card.evolvesFrom, []).append(card)
```

#### 将来の進化先探索

```python
def _future_evolutions(card, evolution_index, max_depth: int = 3) -> list:
    """到達可能な将来の進化先をBFSで収集する。"""
```

- `card is None`なら空リストを返す。
- 起点は`card.name`とする。
- 名前集合をfrontierとして最大3段探索する。
- `cardId`で重複排除する。
- 起点カード自身は結果へ含めない。

実際の進化はBasic→Stage 1→Stage 2の最大2段だが、探索上限には余裕を持たせて3を使う。

#### 進化後の脅威集約

```python
def _evolution_threat_feats(attack_db, evolution_index, card) -> list[float]:
    """進化可能性、最大HP、最大ワザダメージ、ex化の4特徴を返す。"""
```

返却値は次の4要素とする。

```python
[
    has_future_evolution,
    max_evolved_hp / 300.0,
    max_evolved_attack_damage / 300.0,
    will_become_ex,
]
```

- `has_future_evolution`: 候補が1件以上あれば1.0
- `max_evolved_hp`: 全進化候補の最大HP
- `max_evolved_attack_damage`: 全候補の全ワザに含まれる素の最大damage
- `will_become_ex`: `card.ex`または`card.megaEx`が真の候補があれば1.0
- 進化候補がなければ4要素すべて0.0

#### ポケモンスカラーへの統合

`encode_state()`冒頭で`evolution_index`をロードする。
`get_pokemon_scalar()`内で対象ポケモンの`CardData`を取得し、
既存のincoming damage特徴の後ろへ4要素を追加する。

```python
evolution_feats = _evolution_threat_feats(attack_db, evolution_index, card)
return existing_feats + [incoming_max_damage / 300.0] + evolution_feats
```

自分・相手、active・benchのすべてへ同じ処理を適用し、
`get_pokemon_scalar(poke, status, threat_active)`の引数は増やさない。

### 次元の変化

- `POKE_SCALAR_DIM`: 24 → 28
- `global_scalars`: 14のまま
- `concat_dim`: 1422のまま

`PokemonEncoder`の出力は固定の`HIDDEN_DIM`であるため、
ポケモンスカラーの入力次元増加は`concat_dim`へ直接影響しない。

## ラウンド4: スタジアム所有者・ターン内行動・ツール特徴

### 対象ファイル

- `agent/rl_agent.py`
- 対象: `encode_state()`、`get_pokemon_scalar()`、`global_scalars`、
  `POKE_SCALAR_DIM`、`StateEncoder.__init__`

### 解決したかった問題

- スタジアムの有無は認識できても、どちらのプレイヤーが出したか区別できなかった。
- サポート、エネルギー貼付、逃げる、スタジアム使用済みという
  ターン内の行動制約を認識できなかった。
- ポケモンがツールを装備しているか認識できなかった。

### 具体的な仕様

#### スタジアム所有者

場のスタジアムカードの`playerIndex`と`your_idx`を比較する。

```python
is_my_stadium = (
    1.0
    if stadium_card is not None
    and stadium_card.playerIndex == your_idx
    else 0.0
)
```

スタジアムがなければ0.0とする。

#### ターン内行動フラグ

`current`から次の4値をbool→floatで取得する。

```python
supporter_played = float(current.get("supporterPlayed", False))
energy_attached_flag = float(current.get("energyAttached", False))
retreated_flag = float(current.get("retreated", False))
stadium_played_flag = float(current.get("stadiumPlayed", False))
```

`global_scalars`末尾へ次の順で追加する。

```text
is_my_stadium
supporter_played
energy_attached_flag
retreated_flag
stadium_played_flag
```

#### ツール装備フラグ

`get_pokemon_scalar()`の進化特徴の後ろへ次の1要素を追加する。

```python
has_tool = 1.0 if poke.get("tools") else 0.0
```

### 次元の変化

- `POKE_SCALAR_DIM`: 28 → 29
- `global_scalars`: 14 → 19
- `concat_dim`: 1422 → 1427
- 変更後:

```python
POKE_SCALAR_DIM = 2 + N_ENERGY_TYPES + 5 + 4 + 1 + 4 + 1
concat_dim = EMBED_DIM * 3 + HIDDEN_DIM * 4 + 19
```

## ラウンド5: Dragapultヒューリスティックv2

### 対象ファイル

- 新規: `agent/dragapult_agent_v2.py`
- ベース: `agent/dragapult_agent.py`
- 参照のみ: `agent/lucario_v1_agent.py`

既存の`agent/dragapult_agent.py`は変更せず、全文をベースに新規ファイルを作成した。
RL実装はimportせず、torchにも依存しない自己完結したヒューリスティックとした。

### 解決したかった問題

- ATTACHが対象ポケモンやエネルギー種別を見ない一律スコアで、
  複数候補から実質的にランダムな選択をしていた。
- RETREATが相手の攻撃による致死圏を見ず、逃げるコストだけで決まっていた。
- Boss's Orders等の`EFFECT_TARGET`が最低HPだけを見ており、
  ex・Mega exのサイド価値を考慮していなかった。

Behavior Cloningの教師として使用できるよう、特にエネルギー配分の判断を強化した。

### 具体的な仕様

#### 共通ヘルパー

カードDBを初回だけロードする。

```python
_CARD_DB: dict[int, object] | None = None

def _card_data(card_id):
    """all_card_data()をキャッシュし、card IDに対応するCardDataを返す。"""
```

Optionのarea/indexからカードまたはポケモンを解決する。

```python
def _get_card_or_pokemon(obs, area, index, player_index):
    """指定エリアのindex番目を返し、解決失敗時はNoneを返す。"""
```

対象エリアは`HAND`、`ACTIVE`、`BENCH`、`DECK`、`DISCARD`、`PRIZE`、
`STADIUM`、`LOOKING`。例外は捕捉して`None`を返す。

エネルギー要求は多重集合として評価する。

```python
def _remaining_cost(current_energies, extra_type, required) -> int:
    """現在のエネルギーと仮追加1枚で、要求に対して残る不足数を返す。"""
```

処理順は次の通り。

1. 現在のエネルギーへ`extra_type`があれば1枚追加する。
2. 非COLORLESS要求を同色エネルギーと1個ずつマッチし、使用済み分を除く。
3. マッチしなかった色指定要求を不足として数える。
4. 残ったエネルギープールをCOLORLESS要求へ割り当てる。
5. COLORLESS要求を満たせない分を不足へ加える。

#### MAIN_PHASEのATTACH

`opt.area`/`opt.index`から手札のエネルギーカードを取得し、
その`CardData.energyType`を`src_type`とする。
`opt.inPlayArea`/`opt.inPlayIndex`から対象ポケモンを取得する。

対象ポケモンの全ワザについて、貼付前後の不足数を計算する。

```python
before = min(_remaining_cost(current, None, required) for each attack)
after = min(_remaining_cost(current, src_type, required) for each attack)
```

ワザなし・算出不能時のセンチネルは99とする。スコアは次の通り。

```text
対象不明、CardDataなし、ワザなし:
    20.0（このターンにエネルギー貼付済みなら-5.0）

通常:
    40.0
    + 200.0                              if after == 0 and before > 0
    + max(0, before - after) * 30.0       otherwise
    - 30.0                               if before == 99
    + 15.0                               if target is ACTIVE

通常計算で、このターンにエネルギー貼付済み:
    最終スコアから50.0を減算
```

これにより、この1枚でワザが使用可能になる候補を最優先し、
それ以外でも要求エネルギーを実際に減らす候補を高く評価する。

#### MAIN_PHASEのRETREAT

相手アクティブの現在の最大実効ダメージを返すヘルパーを追加する。

```python
def _incoming_threat(obs, my_index) -> float:
    """相手が現在使用可能なワザの最大実効ダメージを返す。"""
```

- 相手アクティブの各ワザを`_remaining_cost(..., None, required) == 0`で絞る。
- 相手アクティブのタイプと自分アクティブの弱点・抵抗を比較する。
- 弱点は2倍、抵抗は-30、下限0。
- 使用可能なワザがなければ0.0。

RETREATのスコアは次の通り。

```text
既にretreat済み:
    -100.0

threat > 0 かつ 自分のactive HP <= threat:
    150.0 - retreat_cost * 5

それ以外:
    30.0 - retreat_cost * 10
```

#### EFFECT_TARGETのサイド価値

相手ポケモンのカード属性からサイド価値を求める。

```python
prize_value = 3.0 if megaEx else 2.0 if ex else 1.0
score = prize_value * 100.0 - hp
```

従来の最低HP比較を、上記スコアが最大の対象を選ぶ比較へ置き換える。
サイド価値を優先し、同じ価値ならHPが低い対象を優先する。

#### 変更しないスコアリング

次のcontext・OptionTypeは旧`dragapult_agent.py`の仕様を維持する。

- `SETUP_ACTIVE`、`SETUP_BENCH`、`SWITCH`
- `TO_BENCH`、`TO_HAND`、`DISCARD`
- `DAMAGE_COUNTER_ANY`、`DISCARD_ENERGY`
- `ATTACK`（`context == 35`）
- MAIN_PHASEの`OPT_ATTACK`、`OPT_ABILITY`、`OPT_EVOLVE`、`OPT_PLAY`、`OPT_END`

### 次元の変化

ヒューリスティックエージェントの新規追加であり、ニューラルネットワークの
`POKE_SCALAR_DIM`、`global_scalars`、`concat_dim`に変更はない。

## 検証方法（全ラウンド共通）

### RLエンコーダ（ラウンド1〜4）

構文チェック:

```bash
cd agent
uv run python -m py_compile rl_agent.py train_ppo.py
```

スモークテストでは`battle_start()`からランダムな合法手を選び、
2ゲーム、最大200〜250ステップ実行する。各選択時に次を確認する。

- `encode_actions()`の末尾次元が`EMBED_DIM = 128`
- `PTCGNet`のlogits shapeが`(1, n_options)`
- 各ラウンドで期待する`global_scalars`とポケモンスカラーのshape
- battleが例外なく終了すること

ラウンド別の主要assert:

```python
# ラウンド2
assert state["global_scalars"].shape[-1] == 14
assert state["my_active_scalar"].shape[-1] == 24
assert state["bench_scalar"].shape[-1] == 24

# ラウンド3
assert rl_agent.POKE_SCALAR_DIM == 28
assert state["my_active_scalar"].shape[-1] == 28
assert state["bench_scalar"].shape[-1] == 28

# ラウンド4
assert rl_agent.POKE_SCALAR_DIM == 29
assert state["global_scalars"].shape[-1] == 19
assert state["my_active_scalar"].shape[-1] == 29
assert state["bench_scalar"].shape[-1] == 29
```

ラウンド3では、カードDBから進化先を持つBasicを探索するユニットテストも実施する。

```python
card_db = rl_agent._load_card_db()
evo_index = rl_agent._load_evolution_index()
found = next(
    (card, rl_agent._future_evolutions(card, evo_index))
    for card in card_db.values()
    if card.basic and rl_agent._future_evolutions(card, evo_index)
)
feats = rl_agent._evolution_threat_feats(
    rl_agent._load_attack_db(), evo_index, found[0]
)
assert feats[0] == 1.0
assert feats[1] > 0.0
```

最小の学習完走確認:

```bash
cd agent
uv run python -u train_ppo.py --episodes 2
```

チェックポイント保存まで例外なく完走することを確認する。
検証で`agent/models/`外に生成された一時的なモデルは削除するが、
既存の学習runやチェックポイントは変更・削除しない。

### Dragapult v2（ラウンド5）

構文チェック:

```bash
cd agent
uv run python -m py_compile dragapult_agent_v2.py
```

スモークテストでは同一の`deck.csv`を使い、以下を各数戦実行する。

- `dragapult_agent_v2` vs `dragapult_agent_v2`
- `dragapult_agent_v2` vs 旧`dragapult_agent`

各ゲームを最大400ステップ進め、クラッシュやPython例外がないことを確認する。
選択されたATTACH optionのindexも記録し、常に同じindexへ固定されていないことを確認する。

## 共通制約

- `encode_state()`、`encode_actions()`の公開シグネチャは変更しない。
- 既存のOption特徴量indexは変更せず、新特徴は末尾へ追加する。
- `agent/encode_state.py`と`submission/rl_agent.py`は変更しない。
- `agent/models/`内の既存run・チェックポイントへ触れない。
- `agent/dragapult_agent.py`を含む既存ヒューリスティックは変更しない。
- `dragapult_agent_v2.py`を`main.py`やPPO対戦相手プールへ登録しない。
- RL特徴量追加では報酬設計を変更しない。
