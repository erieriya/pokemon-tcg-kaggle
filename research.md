# PTCG AI Battle Challenge — Research Notes

> Kaggle競技: The Pokémon Company - PTCG AI Battle Challenge  
> 作成日: 2026-06-17  
> 参照: [Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) / [Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)

---

## 1. 競技概要

### 構成
| カテゴリ | 期間 | 賞金 | 評価軸 |
|---------|------|------|--------|
| Simulation Category | 2026/6/16〜8/17 | なし（Strategy評価の前提） | ラダースコア（Skill Rating） |
| Strategy Category | 2026/6/16〜9/14 | 優勝 $50,000 / 準優勝 $30,000 | 安定性・デッキ設計・戦略レポート |
| Finals（Top 8） | 2026年後半 | $3,000 Google Cloud/人 | 総合評価 |

### 主催・後援
- 主催: The Pokémon Company + Matsuo Institute + HEROZ  
- 後援: Google, Google Cloud, NVIDIA

### 参加方法の注意
- ルールへの同意（Webから）が必要→その後ファイルDL可能
- 1日最大5回までSubmission

---

## 2. シミュレータ環境（cg / cabt Engine）

### 提供ファイル一覧
```
sample_submission/
  main.py         # エントリーポイント（要改造）
  deck.csv        # デッキリスト（60枚）
  cg/
    __init__.py
    api.py        # ゲームAPIメインクラス
    game.py       # ゲームロジック
    sim.py        # シミュレーション実行
    utils.py      # ユーティリティ
    cg.dll        # Windowsバイナリ
    libcg.so      # Linux共有ライブラリ（C製エンジン）
```

### エージェントの動作原理
毎ターン、エージェントは以下を受け取る：
- **observation**: ゲームログ + ボード状態（自分の手札/場/相手の場など）
- **legal_options**: 現在選択可能な行動のリスト（インデックス付き）

エージェントは**選択した行動のインデックスを返す**だけでよい。

### cabt Engine 型定義（API仕様より）

#### Enum 一覧
```python
EnergyType:  COLORLESS=0, GRASS=1, FIRE=2, WATER=3, LIGHTNING=4,
             PSYCHIC=5, FIGHTING=6, DARKNESS=7, METAL=8, DRAGON=9,
             RAINBOW=10, TEAM_ROCKET=11

CardType:    POKEMON=0, ITEM=1, TOOL=2, SUPPORTER=3, STADIUM=4,
             BASIC_ENERGY=5, SPECIAL_ENERGY=6

OptionType:  NUMBER=0, YES=1, NO=2, CARD=3, TOOL_CARD=4, ENERGY_CARD=5,
             ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10,
             DISCARD=11, RETREAT=12, ATTACK=13, END=14, SKILL=15,
             SPECIAL_CONDITION=16

SelectType:  MAIN=0, CARD=1, ATTACHED_CARD=2, CARD_OR_ATTACHED_CARD=3,
             ENERGY=4, SKILL=5, ATTACK=6, EVOLVE=7, COUNT=8, YES_NO=9,
             SPECIAL_CONDITION=10

AreaType:    DECK=1, HAND=2, DISCARD=3, ACTIVE=4, BENCH=5, PRIZE=6,
             STADIUM=7, ENERGY=8, TOOL=9
```

#### データクラス
```python
Card:        id(int), serial(int), playerIndex(int)
Pokemon:     id, serial, hp, maxHp, appearThisTurn(bool),
             energies(list[EnergyType]), energyCards(list[Card]),
             tools(list[Card]), preEvolution(list[Card])
PlayerState: active(list[Pokemon|None]),  # 0 or 1 items
             bench(list[Pokemon]),         # max 5
             hand(list[Card]|None),        # opponent: None
             prize(list[Card|None]),       # face-down = None
             deckCount, handCount, benchMax(int),
             poisoned, burned, asleep, paralyzed, confused(bool)
State:       turn, yourIndex, firstPlayer(int),
             players(list[PlayerState]), stadium(list[Card]),
             supporterPlayed, stadiumPlayed, energyAttached, retreated(bool),
             result(int)
Option:      type(OptionType), attackId, cardId, serial,
             inPlayArea, inPlayIndex, count, energyIndex(int)
SelectData:  type(SelectType), context(SelectContext),
             option(list[Option]), minCount, maxCount(int),
             remainDamageCounter, remainEnergyCost(int)
Observation: select(SelectData|None), current(State|None),
             logs(list[Log])
CardData:    cardId, hp, retreatCost(int), name(str),
             cardType(CardType), energyType(EnergyType),
             basic, stage1, stage2, ex, megaEx, tera, aceSpec(bool),
             evolvesFrom(str|None), weakness/resistance(EnergyType|None),
             attacks(list[int])
Attack:      attackId(int), name, text(str), damage(int),
             energies(list[EnergyType])
```

#### Search API（MCTS用）
```python
search_begin(observation, your_deck, your_prize,
             opponent_deck, opponent_prize,
             opponent_hand, opponent_active,
             manual_coin) → SearchState
search_step(search_id, select) → SearchState
search_end() → None
search_release(search_id) → None
```

### Submission形式
```bash
tar -czvf submission.tar.gz *.py *.csv cg/
# main.pyがトップレベルにあること
```

### 評価（Simulation）
- ラダーマッチング（近いスキルレーティングのエージェントと対戦）
- 勝ちでRating↑、負けでRating↓、引き分けはほぼ変化なし
- 8/31までに収束したらFinal。1日5 submission制限

---

## 3. ゲームルール（PTCG Standard 2026）

### 基本セットアップ
- 60枚デッキ（ポケモン / トレーナー / エネルギー）
- 開始時7枚手札、6枚サイドカード（裏向き）
- ベンチ最大5体
- 先攻はドローなし / 攻撃不可（最近のルール）

### ターン構造
1. **ドロー**（1枚）
2. **メインフェーズ**（好きな順で複数可）
   - ベンチにBasicポケモンをプレイ
   - 進化（Stage1 / Stage2、前のターンに出したものは不可）
   - エネルギーを1枚手からつける
   - トレーナーカード（アイテム：何枚でも / サポート：1枚のみ / スタジアム：1枚）
   - 逃げる（逃げるコスト分のエネルギーを捨てる、1回のみ）
3. **ワザを使う**（エネルギーコスト必要、ターン終了）

### 勝利条件
- 相手を倒してサイドを6枚全部取る
- 相手の場のポケモンがいなくなる
- 相手がデッキ切れ（ドロー不可）

### カードタイプ
| タイプ | 説明 |
|--------|------|
| Basicポケモン | 直接ベンチへ出せる |
| Stage 1 / 2 | 前の進化が場にいれば進化できる |
| ポケモンex | 倒されたらサイド2枚渡す |
| Mega Evolution ex | HP 300-340+、Basicから直接進化 |
| V / VSTAR / VMAX | 旧形式。VSTAR Powerは1試合1回のみ |
| アイテムカード | 1ターンに何枚でも使える |
| サポートカード | 1ターンに1枚のみ |
| スタジアムカード | 場に出すと継続効果、上書きで消える |

### エネルギータイプ（11種）
炎 / 水 / 草 / 雷 / 超 / 闘 / 悪 / 鋼 / 竜 / 無色 / (ふしぎ)

### ダメージ計算
- ワザのダメージ ± 追加効果
- 弱点 × 2 / 抵抗 -30
- ダメカン（10ダメージ = ダメカン1個）

---

## 4. 2026 Standard フォーマット（H-on / 競技環境）

### 規制マーク
- **H / I / J（以降）** → 使用可
- **G以前** → 使用不可（2026/4/10 ローテーション）

### ローテーションアウトした主要カード
| カード | 影響 |
|--------|------|
| Iono | 手札干渉の最強サポートが消滅 |
| Professor's Research | ドロー7枚サポート消滅 |
| Arven | アイテム＋ツール両方サーチ消滅 |
| Nest Ball | Basicポケモンサーチ消滅 |
| Pidgeot ex | 万能サーチエンジン消滅 |
| Counter Catcher | 相手ベンチ呼び出し消滅 |
| Gardevoir ex | 超加速エンジン消滅 |
| Charizard ex | 高火力アタッカー消滅 |
| Gholdengo ex | グッズロックデッキ消滅 |

### 現環境 合法セット
**Scarlet & Violet（H以降）:**
Temporal Forces, Twilight Masquerade, Shrouded Fable, Stellar Crown, Surging Sparks, Prismatic Evolutions, Journey Together, Destined Rivals, Black Bolt, White Flare

**Mega Evolution Series（新シリーズ）:**
Mega Evolution, Phantasmal Flames, Ascended Heroes, Perfect Order

**プロモ:** SVP / MEP Black Star Promos, McDonald's 2024

---

## 5. 2026年メタデッキ分析

### Tier 1

#### Dragapult ex / Dusknoir（最強格）
**戦略:** 序盤はBudewで時間を稼ぎ、Dragapultを複数展開。DracloakのRecon Directiveでドロー、Phantom Diveで相手全体を削りつつDusknoir/DusclopsのCursed Blast（サイド1枚と引き換えに13/5ダメカン配布）でKOを狙う。

**主要カード:**
| カード | 枚数目安 | 役割 |
|--------|---------|------|
| Dragapult ex | 3-4 | メインアタッカー（Phantom Dive: 100+60ベンチ × 2） |
| Drakloak | 3 | 進化サポート + Recon Directiveドロー |
| Dreepy | 4 | 基礎進化前 |
| Dusknoir | 1-2 | Cursed Blast（サイド消費で13ダメカン） |
| Dusclops | 1 | Cursed Blast（5ダメカン、軽量版） |
| Duskull | 2 | Dusknoir進化前 |

**強み:** ローテ後も主要パーツが残存、ベンチダメージで複数同時KO  
**弱み:** Path to the Peak（特性ロック）に弱い、Lily's Clefairy exでDragon弱点操作される

---

#### N's Zoroark ex（Tier 1-2）
**戦略:** N's ZoroarkのCopy Attack（ベンチのNポケモンのワザをコピー）でフレキシブルな打点。ドロー効果つき。

**弱み:** イオナ（手札干渉）の代替が限られた現環境では比較的安定

---

### Tier 2

#### Teal Mask Ogerpon / Mega Evolution Meganium
**戦略:** MeganiumのAbilityで草ポケモンのダメージ2倍。Teal Danceで複数エネルギー加速。

**強み:** Charizard exの退場で草タイプへの圧力が減少

---

#### Mega Lucario ex / Hariyama
**戦略:** Mega LucarioのAbility（Boss's Orders効果）内蔵、闘タイプで高打点。

**弱み:** Lily's Clefairy exによる弱点操作で対策される

---

#### Alakazam / Dundunsparce
**戦略:** ダメカン蓄積でAlakazamの大技を解禁。単サイドデッキとして最強クラス。

---

#### Team Rocket Disruption
**戦略:** Honchkrow等による手札干渉 + Team Rocket系アタッカーで相手のテンポを奪う。

**強み:** ローテ後に汎用ドローが激減 → 手札干渉がより刺さる環境に

---

### Mega Evolution Series の新デッキ

#### Mega Starmie ex / Froslass（日本大会で好成績）
**戦略:** Jet Blow（アクティブ120 + ベンチ50）でベンチ削り。Risky Ruins（ベンチに出た時70HP以下即KO）との相性◎。FroslassラインやMunkidoriでダメカン追加。

**強み:** 2-2マップが取りやすい、複数同時KOで一気にサイドを稼げる  
**採用検討**: ベンチスニーパーはAI学習的に相性良い（ターゲット選択の判断が学習しやすい）

#### Mega Lopunny ex
**戦略:** Gale Thrustで230ダメージをスパム。DudunsparceとAbraでコンシスタンシー確保。Regionalで優勝実績あり。

---

### 競技AI向けデッキ選択の考察

| デッキ | AI学習難度 | 戦略的深さ | 現メタ強さ |
|--------|-----------|-----------|-----------|
| Dragapult ex/Dusknoir | 中（ベンチ分配の最適化） | 高 | Tier 1 |
| Mega Starmie ex/Froslass | 低（ベンチスナイプが明快） | 中 | Tier 2 |
| N's Zoroark ex | 高（コピー対象選択） | 高 | Tier 1-2 |
| Team Rocket Box | 高（手札干渉の応用判断） | 高 | Tier 2 |

**推奨**: まず Dragapult ex/Dusknoir で実装し、メタが固まったら Mega Starmie ex/Froslass を試す。

---

### 重要スタジアム・グッズ・サポート（現環境）

#### 代替ドロー（Ionaの代替）
- **N** (Journey Together): Ionaの類似、ゲーム序中盤の手札干渉
- **Unfair Stamp** (ACE SPEC): 相手の手札を削れる

#### エネルギー加速
- **Ignition Energy**: 進化ポケモンに無色3個分
- **Prism Energy**: Basicに全タイプとして機能
- **Crispin**: 基本エネルギー2種サーチ

#### サーチ・一貫性
- **Hilda**: スペシャルエネルギー + 進化ポケモンを手札に
- **Team Rocket's Watchtower**: 無色ポケモンの特性消去スタジアム
- **Jellicent ex**: グッズ・ツール使用ブロック

#### ダメージ増幅
- **Lillie's Clefairy ex**: Dragonタイプに超弱点を付与（Dragapult対策）
- **Munkidori**: Psychic Embraceデッキでベンチダメージ軽減

---

## 6. 強化学習アプローチの考察

### ゲームの特性（RLの難しさ）
| 特性 | 詳細 |
|------|------|
| 不完全情報 | 相手の手札・デッキ内容が見えない |
| 大規模状態空間 | 2000枚カール × 盤面配置 × ダメカン |
| 可変行動空間 | ターンごとに合法手が大きく変わる |
| スパース報酬 | 試合終了時のみ勝敗判定 |
| 確率的遷移 | ドロー、シャッフル |

### アルゴリズム選択指針

#### PPO（Proximal Policy Optimization）+ Self-Play ← **推奨**
- カードゲームでの実績多数（Big 2, Poker, Hearthstone等）
- Actor-Critic + GAEで分散削減
- エントロピー正則化（β=0.05推奨）でover-determinismを防ぐ
- 現在ポリシーへの自己対戦が固定カリキュラムより強い
- 学習速度でDQNより優れる

#### Counterfactual Regret Minimization（CFR / MCCFR）
- 不完全情報ゲームのナッシュ均衡に理論的収束
- Deep CFR: 状態空間が大きい場合にNN近似
- ポケモンTCGの状態空間ではおそらく計算コスト過大
- ポーカーや麻雀で実績あり

#### MCTS（Monte Carlo Tree Search）
- 完全情報ゲームで強いが、隠情報下では確率的サンプリング必要
- 合法手が多いターン（特に手札が多い序盤）で分岐爆発
- A2CやDQNの指導として組み合わせるのが現実的

#### Hybrid（推奨候補）
- PPO + MCTS rollout（AZ-style）
- 状態価値評価にNNを使い、各ターンで数回MCTSシミュレーション
- 計算コストとのトレードオフを検討

### 状態表現の設計

```python
observation = {
    # 自分の情報
    "hand": [card_id, ...],              # 手札（非公開情報だが自分は見える）
    "active_pokemon": {
        "card_id": int,
        "hp_remaining": int,
        "damage_counters": int,
        "attached_energy": {type: count},
        "status_condition": str,         # burn/poison/sleep/paralysis/confusion
        "abilities_active": [bool],
    },
    "bench": [...],                      # 最大5体
    "prize_cards_remaining": int,        # 残りサイド
    "discard_pile": [card_id, ...],
    
    # 相手の公開情報
    "opponent_active": {...},            # 相手のアクティブ（HP等は見える）
    "opponent_bench": [...],             # 相手ベンチ（公開情報のみ）
    "opponent_prize_cards_remaining": int,
    
    # 場の情報
    "stadium": card_id,
    "turn_number": int,
    "is_first_player": bool,
}

legal_actions = [
    # インデックス付きで提供される
    {"index": 0, "type": "play_basic", "card": card_id, "position": 0-5},
    {"index": 1, "type": "evolve", "from": card_id, "to": card_id},
    {"index": 2, "type": "attach_energy", "card": energy_id, "target": position},
    {"index": 3, "type": "play_trainer", "card": card_id, ...},
    {"index": 4, "type": "retreat", "new_active": position},
    {"index": 5, "type": "attack", "attack_index": 0-3},
    {"index": 6, "type": "pass"},
]
```

### ニューラルネットワーク設計案

```
Input:
  - カード埋め込み（card_id → 128dim embedding）
  - 手札: Self-Attention Pooling over card embeddings
  - 場のポケモン: 構造化特徴量（HP比率, エネルギー数, タイプetc）
  - ゲーム状態スカラー（サイド枚数, ターン数, etc）

Policy Head:
  - 合法手のembeddingと状態embeddingのAttention
  - softmax over legal actions

Value Head:
  - FC layers → scalar（勝率予測 0~1）
```

### 報酬設計（Reward Shaping）

スパース報酬（勝敗のみ）では学習が遅い。中間報酬を追加する:

```python
def compute_reward(prev_state, curr_state, done, winner):
    if done:
        return 1.0 if winner == 0 else -1.0

    reward = 0.0

    # サイドカードを取った (÷6 で正規化)
    my_prizes_prev = prev_state["players"][0]["prize"].__len__()
    my_prizes_curr = curr_state["players"][0]["prize"].__len__()
    if my_prizes_curr < my_prizes_prev:
        reward += (my_prizes_prev - my_prizes_curr) / 6.0 * 0.5

    # 相手にサイドを取られた
    opp_prizes_prev = prev_state["players"][1]["prize"].__len__()
    opp_prizes_curr = curr_state["players"][1]["prize"].__len__()
    if opp_prizes_curr < opp_prizes_prev:
        reward -= (opp_prizes_prev - opp_prizes_curr) / 6.0 * 0.5

    # 相手のHPを削った（ダメカンの増加）
    opp_hp_prev = _get_damage(prev_state["players"][1]["active"])
    opp_hp_curr = _get_damage(curr_state["players"][1]["active"])
    reward += max(0, opp_hp_curr - opp_hp_prev) / 300.0 * 0.1

    return reward
```

**注意**: 過度な中間報酬はサブ最適（例: KOせずダメカン乗せに固執）を生む。  
最終的には純粋な勝敗報酬の比率を上げていくアニーリングが有効。

### プライズマッピング（Prize Mapping）戦略

ポケモンTCGではKOで得られるサイド枚数が異なる:
| ポケモン種別 | 倒した時のサイド枚数 |
|------------|------------------|
| 通常ポケモン | 1枚 |
| ポケモンex / V | 2枚 |
| VMAX / Mega ex | 3枚 |

**最適なプライズマップの例:**
- 2-2-2: 相手のex3体を倒す（合計6枚）
- 1-1-2-2: 小物2体 + ex2体
- 1-1-1-1-2: 積極的に小物で削り、最後にex

**戦略的含意:**
- ベンチにexを無駄に並べると相手がサイドを取りやすい
- 単サイドアタッカーでexを狩ると「プライズトレード有利」
- Dusknoir/DusclopsのCursed Blastはサイド1枚消費→敵exを狩ると+1トレード

### 学習戦略

1. **ランダムエージェントとの対戦でウォームアップ**（初期は合法手のランダム選択）
2. **PPO自己対戦**（current policy vs. 少し前のpolicy）
3. **リーグ対戦**（過去の強いバージョンと定期的に対戦、多様性確保）
4. **エントロピーアニーリング**（学習後期に確実性を上げる）
5. **報酬アニーリング**: 初期は中間報酬を多めに、後期は勝敗報酬のみに絞る

---

## 7. デッキ構築戦略

### 60枚の配分目安
| カテゴリ | 枚数 |
|---------|------|
| ポケモン | 12-18 |
| トレーナー | 28-36 |
| エネルギー | 8-16 |

### AIによるデッキ最適化アプローチ

#### アプローチ1: 固定デッキ + 行動最適化
- 既存の競技実績デッキ（Dragapult ex/Dusknoir）を使用
- デッキは固定してエージェントの意思決定のみ最適化
- 実装が最もシンプルで効果が出やすい

#### アプローチ2: 共進化（Co-evolution）
- デッキとエージェントを同時に進化
- 遺伝的アルゴリズムでデッキを変異・交叉
- 評価: そのデッキを使ったエージェントの勝率
- 探索空間が大きいため学習に時間がかかる

#### アプローチ3: RL-based deckbuilding
- デッキ構築もMDPとして定式化
- 各カードの採用をaction（二値）として学習
- 報酬: 完成デッキで戦った勝率

### Dragapult ex/Dusknoir サンプルデッキリスト（2026 H-on）

```
ポケモン (17枚)
4 Dreepy
3 Drakloak
3 Dragapult ex
2 Duskull
1 Dusclops
1 Dusknoir
1 Fezandipiti ex
1 Munkidori
1 Budew（序盤時間稼ぎ）

トレーナー (35枚)
4 Hilda
3 Crispin
3 N
2 Boss's Orders
2 Unfair Stamp
4 Ultra Ball
4 Rare Candy
3 Nest Ball（代替必要かも）
2 Technical Machine: Devolution
2 Counter Catcher（代替必要かも）
3 Path to the Peak（メタカード）
3 PokéStop（ドロースタジアム）

エネルギー (8枚)
4 Basic Fire Energy
4 Basic Psychic Energy
```

### デッキの一貫性（Consistency）の計算
```python
# ハイパージオメトリック分布でカードをドローできる確率
from scipy.stats import hypergeom

def prob_draw_at_least_n(deck_size, card_count, draw_count, n):
    # deck_size: デッキ総枚数(60)
    # card_count: 目当てのカードの枚数
    # draw_count: 何枚引くか（初手7枚 + サポート等）
    # n: 少なくともn枚欲しい
    rv = hypergeom(deck_size, card_count, draw_count)
    return 1 - rv.cdf(n-1)

# 例: 4積みのカードを初手7枚で1枚以上引く確率
print(prob_draw_at_least_n(60, 4, 7, 1))  # ≈ 0.40
# サポートで7枚引けた後なら（合計14枚見た）
print(prob_draw_at_least_n(60, 4, 14, 1))  # ≈ 0.66
```

---

## 8. 実装ロードマップ

### Phase 1: ベースライン（1-2週間）
- [ ] 競技ルール同意 → データDL（Webから手動）
- [ ] `cg/api.py`を読んでObservation/Action構造を把握
- [ ] ランダムエージェント実装 & テスト
- [ ] Dragapult ex/Dusknoir固定デッキで提出してスコア確認

### Phase 2: RLエージェント（2-4週間）
- [ ] 状態エンコーダー設計（カード埋め込み + Attention）
- [ ] PPO実装（または stable-baselines3 活用）
- [ ] 自己対戦ループ構築
- [ ] 勝率改善のモニタリング

### Phase 3: 改善（4-8週間）
- [ ] リーグ対戦で多様な相手と対戦
- [ ] デッキチューニング（遺伝的アルゴリズム or ベイズ最適化）
- [ ] メタゲーム分析（どのデッキに弱いか把握）
- [ ] 相手デッキ推定モジュール追加

### Phase 4: Strategyレポート（8/17〆切前）
- [ ] アーキテクチャ説明
- [ ] デッキ選択理由と対メタ分析
- [ ] アブレーション研究（何が効いたか）

---

## 9. 参考リソース

### 競技
- [Simulation Competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
- [Strategy Competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)

### PTCG メタ情報
- [Limitless TCG - Dragapult/Dusknoir デッキ統計](https://play.limitlesstcg.com/decks/dragapult-dusknoir?format=standard&rotation=2026)
- [Wargamer - 2026 Post-Rotation Best Decks](https://www.wargamer.com/pokemon-trading-card-game/post-rotation-best-decks-2026)
- [Professor's Research - Standard 2026 Format](https://www.professorsresearch.com/formats/standard-2026)
- [Bulbapedia - 2026-27 Standard Format](https://bulbapedia.bulbagarden.net/wiki/2026-27_Standard_format_(TCG))

### RL・AI論文
- [Self-Play RL under Imperfect Information (Big 2)](https://arxiv.org/html/2605.28863) — PPO推奨
- [Survey on Self-Play Methods in RL](https://arxiv.org/pdf/2408.01072)
- [Deep CFR for Trading Card Games (AI 2022)](https://link.springer.com/chapter/10.1007/978-3-031-22695-3_11)
- [RLCard Toolkit](https://rlcard.org/)
- [Pokemon RL (poke_RL GitHub)](https://github.com/leolellisr/poke_RL)
- [Can Large Language Models Master Complex Card Games?](https://arxiv.org/pdf/2509.01328) — LLMはRLより弱い
- [Policy-Based RL in Imperfect Information Card Game](https://doi.org/10.3390/app15042121)
- [Mastering Board Games with Language Models](https://arxiv.org/pdf/2412.12119)

### シミュレータ（参考実装）
- [PokemonTCGP-BattleSimulator](https://github.com/AngelFireLA/PokemonTCGP-BattleSimulator)
- [ryuu-play](https://github.com/keeshii/ryuu-play)

---

## 10. LLM アプローチについての考察

### LLM の限界
- 長期的な計画（Long-horizon planning）が苦手
- 隠情報の管理が不得意（相手の手札推定など）
- MCTSやRLと比べて競技カードゲームではパフォーマンス低い
- ただし戦略レポート（Strategy Category）の記述には有用

### ハイブリッドアプローチの可能性
- **LLM + RL**: LLMで大まかな方針を生成し、RLでfine-tuning
- **LLM as judge**: LLMでデッキのシナジーを評価してデッキ候補を絞り込む
- **RAG + 戦略DB**: 過去の試合ログをRAGで参照、試合中の戦略選択に活用

### 現実的な推奨
ポケモンTCGの競技AIとして最高パフォーマンスを目指すなら:
1. **PPO + Self-play** をベースに（実証済み最強）
2. Search API (MCTS) を組み合わせ（時間があれば）
3. LLMはデッキ分析・Strategyレポート作成に使用
