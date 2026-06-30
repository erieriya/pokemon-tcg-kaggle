# ポケモンTCG RL特徴量 追加候補調査

> 作成: 2026-07-01  
> 目的: PTCGNetへの入力特徴量の網羅的リストアップ。この中から実装コスト対効果を見て
> 段階的に追加していく。現在の実装状況は `rl_agent.py` の `encode_state` を参照。

---

## ゲームAPIで取得可能な情報源

### CardData
`cardId, name, cardType, retreatCost, hp, weakness, resistance, energyType, basic, stage1, stage2, ex, megaEx, tera, aceSpec, evolvesFrom, skills, attacks`

### Pokemon (インゲーム)
`id, serial, hp, maxHp, appearThisTurn, energies, energyCards, tools, preEvolution`

### PlayerState
`active, bench, benchMax, deckCount, discard, prize, handCount, hand, poisoned, burned, asleep, paralyzed, confused`

### State
`turn, turnActionCount, yourIndex, firstPlayer, supporterPlayed, stadiumPlayed, energyAttached, retreated, result, stadium, looking, players`

### Attack
`attackId, name, text, damage, energies`

### Skill
`name, text`

### SelectData
`type, context, minCount, maxCount, option, remainDamageCounter, remainEnergyCost, contextCard, effect, deck`

---

## 現在エンコード済みの特徴量 (rl_agent.py時点)

### 入力テンソル
- `hand_ids`: 自分手札カードID (最大20枚)
- `my_active_id/scalar`, `opp_active_id/scalar`
- `bench_ids/scalar/mask`, `opp_bench_ids/scalar/mask` (各最大5枠)
- `stadium_id`
- `opp_discard_ids/mask` (最大60枚)
- `global_scalars`: 24次元

### POKE_SCALAR_DIM = 49 (2026-07-01更新)
| 特徴量 | 次元 |
|---|---|
| HP比率 (hp/maxHp) | 1 |
| 受けダメージ (dmg/300) | 1 |
| 付きエネルギータイプ別 | 12 |
| 状態異常5種 | 5 |
| retreatCost, ex, megaEx, stage | 4 |
| 相手アクティブからの最大被ダメージ | 1 |
| 将来の進化先 [has_next_stage, next_hp/300, next_max_dmg/300, next_is_ex] | 4 |
| ツール有無 | 1 |
| ability有無, EX免疫, ダメカン免疫エネルギー | 3 |
| カード自身のエネルギータイプ1-hot | 12 |
| 弱点タイプ (正規化) | 1 |
| 耐性タイプ (正規化) | 1 |
| can_attack_now | 1 |
| max_outgoing_damage/300 | 1 |
| appear_this_turn | 1 |

### global_scalars = 24次元
| 特徴量 | |
|---|---|
| 自分/相手サイド残り各/6 | 2 |
| ターン数/50 | 1 |
| 自分/相手デッキ枚数/60 | 2 |
| 自分/相手ベンチ数/5 | 2 |
| benchMax/5 | 1 |
| スタジアム有無 | 1 |
| 自分/相手手札枚数/20 | 2 |
| 先攻か | 1 |
| 自分/相手 ready_attacker数/6 | 2 |
| is_my_stadium | 1 |
| supporter/energy/retreat/stadium_played フラグ | 4 |
| can_ko_opp_active, opp_can_ko_me | 2 |
| prize_diff/6 | 1 |
| turnActionCount/10 | 1 |
| opp_discard_ex_count/5 | 1 |

### encode_actions (per-option特徴量)
- OptionType one-hot (17次元)
- attack damage, count
- attack energy requirement (12次元)
- attack total cost
- attach source energy type (12次元)
- attach target current energy (12次元)
- attach target is active
- attach unlocks attack
- attach remaining energy count
- attack effective damage (vs weakness/resistance)
- attack would KO

---

## 追加候補一覧

### A) per-Pokémonスカラー (POKE_SCALAR_DIM変更 → 新BC pretrain必要)

| # | 特徴名 | 次元 | 取得方法 | 重要度 | 難易度 |
|---|---|---|---|---|---|
| P1 | HP実数正規化 (hp/300) | 1 | `Pokemon.hp / 300` | ★★★ | easy |
| P2 | MaxHP実数正規化 (maxHp/300) | 1 | `Pokemon.maxHp / 300` | ★★ | easy |
| P3 | サイド価値 (ex=2, megaEx=3, 通常=1) | 1 | CardData.ex/megaEx | ★★★ | easy |
| P4 | Teraフラグ | 1 | `CardData.tera` | ★★ | easy |
| P5 | ACE SPECフラグ | 1 | `CardData.aceSpec` | ★ | easy |
| P6 | Basic/Stage1/Stage2 one-hot | 3 | `basic/stage1/stage2` | ★★ | easy |
| P7 | 特殊エネルギー枚数 | 1 | `energyCards` の SPECIAL_ENERGY | ★★★ | easy |
| P8 | ツール枚数 | 1 | `len(Pokemon.tools)` | ★★ | easy |
| P9 | 逃げエネ不足数 | 1 | `retreatCost - len(energies)` | ★★★ | easy |
| P10 | 今すぐ逃げられるか | 1 | `len(energies) >= retreatCost` | ★★★ | easy |
| P11 | 最小ワザ要求までの不足エネ | 1 | 各Attackと現在エネを比較 | ★★★ | easy |
| P12 | 次の1エネで解放される最大打点 | 1 | 各エネタイプ仮付けでAttack評価 | ★★★ | medium |
| P13 | 最高素点打点/300 | 1 | CardData.attacks → Attack.damage max | ★★ | easy |
| P14 | 最高実効打点/300 (弱点込み) | 1 | weakness/resistance適用後 | ★★★ | medium |
| P15 | 相手アクティブへの不足打点/300 | 1 | opp.hp - max_effective_dmg | ★★★ | easy |
| P16 | ベンチ狙撃テキスト有無 | 1 | Attack.text キーワード "bench" | ★★ | medium |
| P17 | ダメカン配置テキスト有無 | 1 | Skill/Attack.text "damage counter" | ★★ | medium |
| P18 | ドロー/サーチ能力有無 | 1 | Skill/Attack.text "draw/search" | ★★★ | medium |
| P19 | エネ加速能力有無 | 1 | Skill/Attack.text "attach/energy" | ★★★ | medium |
| P20 | 入れ替え/逃げ補助能力有無 | 1 | Skill/Attack.text "switch/retreat" | ★★ | medium |
| P21 | 状態異常付与能力有無 | 1 | Skill/Attack.text 状態異常キーワード | ★★ | medium |
| P22 | 自分状態異常で攻撃不能リスク | 1 | asleep/paralyzed/confused (アクティブのみ) | ★★★ | easy |
| P23 | 相手弱点を突けるタイプか | 1 | 自身energyType == 相手weakness | ★★★ | easy |
| P24 | 相手抵抗で軽減されるタイプか | 1 | 自身energyType == 相手resistance | ★★ | easy |
| P25 | preEvolutionの枚数/深さ | 1 | `len(Pokemon.preEvolution) / 2` | ★★ | easy |
| P26 | 進化深さが最大か (これ以上進化なし) | 1 | 次の進化先がないか | ★★ | easy |
| P27 | このターン登場したか (appearThisTurn) | 1 | 既に実装済み | - | - |

**注**: ポケカの「この場面ではこうなので、この方法をとります」的な判断を特徴量化するなら：
- P3 (サイド価値) + P15 (不足打点) → 「これをKOすれば相手がサイドXを取る → 逃げるべきか」
- P12 (1エネで解放) → 「エネを貼ればワザが打てる = 今ターン貼るべき」
- P22 (攻撃不能リスク) → 「麻痺/眠りなら他のポケモンに入れ替えるべき」
といった暗黙の戦術ルールが自然に学習されやすくなる。

---

### B) グローバルスカラー追加 (global_scalars, モデルアーキテクチャ変更必要)

| # | 特徴名 | 次元 | 取得方法 | 重要度 | 難易度 |
|---|---|---|---|---|---|
| G1 | 自分捨て札枚数/60 | 1 | `len(my.discard)` | ★★ | easy |
| G2 | 自分捨て札エネルギー枚数 | 1 | discard cardType==ENERGY | ★★★ | easy |
| G3 | 自分捨て札サポート枚数 | 1 | discard cardType==SUPPORTER | ★★ | easy |
| G4 | 自分捨て札ポケモン枚数 | 1 | discard cardType==POKEMON | ★★ | easy |
| G5 | 相手捨て札エネルギー枚数 | 1 | opp discard cardType==ENERGY | ★★ | easy |
| G6 | 手札タイプ構成 (ポケ/グッズ/道具/サポ/スタジアム/エネ/その他) | 7 | hand CardData.cardType集計 | ★★★ | easy |
| G7 | 手札エネルギータイプ構成 | 12 | hand energyType counts | ★★★ | easy |
| G8 | 手札のたねポケモン数 | 1 | hand CardData.basic | ★★★ | easy |
| G9 | 手札の進化カード数 | 1 | hand stage1/stage2 | ★★★ | easy |
| G10 | 手札のサポート数 | 1 | hand cardType SUPPORTER | ★★★ | easy |
| G11 | ベンチ空き数/5 | 1 | `(benchMax - len(bench)) / 5` | ★★★ | easy |
| G12 | 自分場の総残りHP/1800 | 1 | active+bench hp sum | ★★ | easy |
| G13 | 相手場の総残りHP/1800 | 1 | 同上 | ★★ | easy |
| G14 | 自分場の総エネ枚数/30 | 1 | active+bench energies count | ★★★ | easy |
| G15 | 相手場の総エネ枚数/30 | 1 | 同上 | ★★★ | easy |
| G16 | 自分場の特殊エネ総数/10 | 1 | energyCards SPECIAL_ENERGY | ★★ | easy |
| G17 | 相手場の特殊エネ総数/10 | 1 | 同上 | ★★ | easy |
| G18 | 手札からベンチに出せるたね数 | 1 | bench_space と hand basic | ★★★ | easy |
| G19 | 手札から進化できる数 | 1 | options EVOLVEの数 | ★★★ | medium |
| G20 | デッキアウト危険度差 | 1 | deckCount差 (負=自分が先になくなる) | ★★★ | easy |
| G21 | SelectContext one-hot | ~49 | `obs.select.context` | ★★★ | easy |
| G22 | SelectType one-hot | ~11 | `obs.select.type` | ★★★ | easy |
| G23 | remainDamageCounter/30 | 1 | `SelectData.remainDamageCounter` | ★★ | easy |
| G24 | remainEnergyCost/5 | 1 | `SelectData.remainEnergyCost` | ★★ | easy |
| G25 | looking中のカード数/20 | 1 | `len(State.looking)` | ★★★ | easy |
| G26 | 自分/相手の合計ツール数 | 2 | Pokemon.tools 集計 | ★★ | easy |
| G27 | 自分アクティブのサイド価値 (ex/megaEx) | 1 | P3と同じ計算 | ★★★ | easy |
| G28 | 相手アクティブのサイド価値 | 1 | 同上 | ★★★ | easy |

---

### C) per-action特徴量追加 (encode_actions拡張)

| # | 特徴名 | 次元 | 取得方法 | 重要度 | 難易度 |
|---|---|---|---|---|---|
| A1 | 行動対象カードID (embedding) | EMBED_DIM | Option area/index から解決 | ★★★ | easy |
| A2 | 行動対象がActive/Bench/Hand/Discardか | 8 | `Option.area` one-hot | ★★★ | easy |
| A3 | PLAYするカードタイプ | 7 | CardData.cardType one-hot | ★★★ | easy |
| A4 | PLAYがサポートか | 1 | cardType==SUPPORTER | ★★★ | easy |
| A5 | PLAYがスタジアムか | 1 | cardType==STADIUM | ★★ | easy |
| A6 | PLAYがACE SPECか | 1 | CardData.aceSpec | ★★ | easy |
| A7 | ATTACH後の対象最大打点 | 1 | 仮付け後 `_max_affordable_damage` | ★★★ | medium |
| A8 | ATTACH後にKO可能か | 1 | 仮付け後 vs opp HP | ★★★ | medium |
| A9 | ATTACH対象が次ターン倒される危険 | 1 | opp_max_dmg >= target_hp | ★★★ | medium |
| A10 | EVOLVE後HP増加量/300 | 1 | evolved.hp - target.maxHp | ★★★ | medium |
| A11 | EVOLVE後最大打点増加/300 | 1 | evolved attacks max dmg | ★★★ | medium |
| A12 | EVOLVE後ex化か | 1 | evolved CardData.ex | ★★ | easy |
| A13 | EVOLVE対象がappearThisTurnか | 1 | target.appearThisTurn | ★★★ | easy |
| A14 | ATTACK反動/デメリット有無 | 1 | Attack.text "discard/damage self" | ★★ | medium |
| A15 | ATTACKエネ妨害有無 | 1 | Attack.text "discard.*energy" | ★★★ | medium |
| A16 | ATTACKベンチダメージ有無 | 1 | Attack.text "bench" | ★★ | medium |
| A17 | ATTACKドロー/サーチ効果有無 | 1 | Attack.text "draw/search" | ★★ | medium |
| A18 | ATTACK状態異常付与 (5種) | 5 | Attack.text キーワード | ★★ | medium |
| A19 | DAMAGE_COUNTERでKOできるか | 1 | remainDamageCounter vs target.hp/10 | ★★★ | medium |
| A20 | DAMAGE_COUNTER対象HP/300 | 1 | target Pokemon.hp | ★★★ | easy |
| A21 | DISCARDするカードの価値スコア | 1 | ex/energy/supporter等から重み付け | ★★★ | medium |
| A22 | RETREAT後の代替アタッカー最大打点 | 1 | bench → max_affordable_damage | ★★★ | medium |
| A23 | SelectContext one-hot (action側でも持つ) | ~49 | `obs.select.context` | ★★★ | easy |
| A24 | 行動がターン終了 (END) か | 1 | OptionType.END | ★★★ | easy |

---

### D) 新しい入力テンソル候補 (モデルアーキテクチャ変更が大きい)

| # | テンソル名 | 形状 | 取得方法 | 重要度 | 難易度 |
|---|---|---|---|---|---|
| T1 | `my_discard_ids` | (60,) + mask | `my.discard` | ★★★ | easy |
| T2 | `hand_card_type_counts` | (7,) | hand CardData.cardType集計 | ★★★ | easy |
| T3 | `hand_energy_type_counts` | (12,) | hand energyType counts | ★★★ | easy |
| T4 | `looking_ids` | (~20,) + mask | `State.looking` (サーチ中のカードリスト) | ★★★ | easy |
| T5 | `my_active_tool_ids` | (~3,) + mask | `Pokemon.tools` | ★★★ | medium |
| T6 | `opp_active_tool_ids` | (~3,) + mask | 同上 | ★★★ | medium |
| T7 | `my_active_energy_card_ids` | (~10,) + mask | `Pokemon.energyCards` | ★★ | medium |
| T8 | `select_deck_ids` | (~60,) + mask | `SelectData.deck` (サーチ時に見えているデッキ) | ★★★ | medium |

---

## 優先実装ロードマップ (推奨順)

### v7 (現在実装済み, POKE_SCALAR=49, global=24)
- card_energy_type (12), weakness/resistance, can_attack_now, max_outgoing_damage, appear_this_turn
- global: can_ko, opp_can_ko, prize_diff, turnActionCount, opp_discard_ex_count

### v8 候補 (次世代, 新BC pretrain必要)
**高インパクト・低コスト:**
- P3: サイド価値 (ex=2倍) ← ゲームを勝負するうえで最重要
- P9/P10: 逃げエネ不足数 / 今すぐ逃げられるか
- P23/P24: 相手弱点を突けるか / 相手抵抗で軽減か
- G6-G10: 手札タイプ/エネ構成, たね数, サポ数
- G11: ベンチ空き数
- G21/G22: SelectContext/Type one-hot (行動文脈を明示)

**高インパクト・中コスト:**
- A7/A8: ATTACH後の最大打点 / KO可否 (エネ貼り先の精度UP)
- A10/A11: EVOLVE後HP/打点増加
- G23/G24: remainDamageCounter/remainEnergyCost

### v9以降 (大規模追加)
- T1: `my_discard_ids` テンソル追加 (自分の捨て札から状況判断)
- T4: `looking_ids` (サーチ系の選択精度UP)
- D系: 手札タイプ別テンソル
- 戦術ルールの明示化: 「KOすれば相手がサイドX枚取る」→ P3×G28のクロス特徴

---

## 「この場面ではこうなので」的な戦術特徴量の候補

ポケカの実際の判断フロー (ベテランプレイヤーの思考) を特徴量に変換するアイデア:

| 戦術判断 | 必要な特徴量の組み合わせ |
|---|---|
| 「相手アクティブをKOすれば次のサイドを取れる」 | can_ko_opp + opp_active_prize_value |
| 「倒されると2サイド失う危険がある」 | opp_can_ko_me + my_active_prize_value |
| 「今エネを貼れば次ターン攻撃できる」 | 1エネ追加で解放される最大打点(P12) |
| 「逃げた方がいい(今のポケモンは戦えない)」 | P9(逃げエネ不足) + P22(攻撃不能) + P15(不足打点) |
| 「進化すれば場が強くなる」 | A10(HP増) + A11(打点増) + A13(appearThisTurn=逃げ不可) |
| 「デッキアウトで負ける危険」 | G20(デッキ差) + ターン数 |
| 「手札が悪い(アクションが少ない)」 | G10(サポ数) + G8(たね数) + G14(エネ数) |
| 「スタジアムを張れる = 有利効果を得る」 | G16(手札スタジアム数) + G11(現スタ是非) |

---

## 注意点

- `CardData.skills` と `Attack.text` は自然言語なので、キーワードマッチングで固定特徴化する (hard だが効果大)
- `opp.hand` はAPI上 `None` → 相手手札の詳細は直接取得不可 (handCount + logs からの推定のみ)
- `prize` は `Card | None` → Noneをmaskし、見えているカードだけID化
- `State.looking` はサーチ系のアクションでのみ現れる特殊フィールド → maskが必要
- POKE_SCALAR_DIM変更 = モデルアーキテクチャ変更 = 新BC pretrain必須 (約21分)
- global_scalars次元変更も同様に新BC pretrain必須 (concat_dimの変更)
- SelectContext/Type one-hotは次元数が多い (~49+11) ので、global全体が100次元超になりうる

---

## 現在のPOKE_SCALAR_DIM/global_scalarsへの追加インデックス

新特徴量を追加する際の参考:
```python
# 現在の内訳 (rl_agent.py get_pokemon_scalar の return 順)
# index 0: hp_ratio
# index 1: dmg/300
# index 2-13: energy_vec (N_ENERGY_TYPES=12)
# index 14-18: status (5)
# index 19-22: card_feats [retreatCost, ex, megaEx, stage]
# index 23: incoming_max_damage/300
# index 24-27: evolution_feats [has_next, next_hp, next_dmg, next_is_ex]
# index 28: has_tool
# index 29: has_ability
# index 30: is_ex_damage_immune
# index 31: has_damage_counter_immune_energy
# index 32-43: card_energy_type_vec (12) ← 2026-07-01追加
# index 44: weakness_val ← 2026-07-01追加
# index 45: resistance_val ← 2026-07-01追加
# index 46: can_attack_now ← 2026-07-01追加
# index 47: max_outgoing/300 ← 2026-07-01追加
# index 48: appear_this_turn ← 2026-07-01追加
```
