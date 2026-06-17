"""
Dragapult ex / Dusknoir デッキ専用ヒューリスティックエージェント

コア戦略:
1. Phantom Dive (200点 + ベンチ6カウンター最適分配)
2. Dusknoir Cursed Blast (13カウンター → KO狙い確定)
3. Dusclops Cursed Blast (5カウンター → 補助打点)
4. Unfair Stamp / Boss's Orders で終盤詰め
"""

import os
import random
from typing import Optional

# ─── カードID定数 ───────────────────────────────────────────────
DREEPY        = 119
DRAKLOAK      = 120
DRAGAPULT_EX  = 121
DUSKULL       = 131
DUSCLOPS      = 132
DUSKNOIR      = 133
FEZANDIPITI_EX = 140
BUDEW         = 235

ULTRA_BALL         = 1121
RARE_CANDY         = 1079
BUDDY_BUDDY_POFFIN = 1086
DUSK_BALL          = 1102
BUG_CATCHING_SET   = 1094
POKEGEAR           = 1122
UNFAIR_STAMP       = 1080  # ACE SPEC ×1のみ
HILDA              = 1225
CRISPIN            = 1198
BOSSS_ORDERS       = 1182
JUDGE              = 1213

FIRE_ENERGY    = 2
PSYCHIC_ENERGY = 5

# Dusknoir Cursed Blast: 13カウンター=130dmg → 自爆
# Dusclops Cursed Blast:  5カウンター= 50dmg → 自爆
DUSKNOIR_CURSED_DMG = 130
DUSCLOPS_CURSED_DMG = 50

# ─── SelectContext 値 (cg.api.SelectContext と対応) ─────────────
CTX_MAIN               = 0
CTX_SETUP_ACTIVE       = 1
CTX_SETUP_BENCH        = 2
CTX_SWITCH             = 3
CTX_TO_ACTIVE          = 4
CTX_TO_BENCH           = 5
CTX_TO_FIELD           = 6
CTX_TO_HAND            = 7
CTX_DISCARD            = 8
CTX_TO_DECK            = 9
CTX_DAMAGE_COUNTER     = 13
CTX_DAMAGE_COUNTER_ANY = 14
CTX_EVOLVE_FROM        = 18
CTX_EVOLVE_TO          = 19
CTX_ATTACH_FROM        = 21
CTX_LOOK               = 24
CTX_EFFECT_TARGET      = 25
CTX_DISCARD_ENERGY     = 26
CTX_ATTACK             = 35
CTX_EVOLVE             = 37
CTX_DRAW_COUNT         = 38

# ─── OptionType 値 ───────────────────────────────────────────────
OPT_NUMBER   = 0
OPT_YES      = 1
OPT_NO       = 2
OPT_CARD     = 3
OPT_TOOL     = 4
OPT_ENERGY_C = 5
OPT_ENERGY   = 6
OPT_PLAY     = 7
OPT_ATTACH   = 8
OPT_EVOLVE   = 9
OPT_ABILITY  = 10
OPT_DISCARD  = 11
OPT_RETREAT  = 12
OPT_ATTACK   = 13
OPT_END      = 14
OPT_SKILL    = 15


# ─── ユーティリティ ──────────────────────────────────────────────

def _g(obj, key, default=None):
    """dict / dataclass から安全にフィールドを取得"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _opt_type(opt) -> int:
    return _g(opt, "type", OPT_END)


def _card_id(obj) -> int:
    return _g(obj, "id", 0) or _g(obj, "cardId", 0) or 0


def _remaining_hp(pokemon) -> int:
    if pokemon is None:
        return 9999
    return max(0, _g(pokemon, "hp", 100))


def _max_hp(pokemon) -> int:
    if pokemon is None:
        return 0
    return _g(pokemon, "maxHp", 100)


def _all_field_pokemon(state, player_idx: int) -> list:
    """Active + Bench を結合して返す"""
    if state is None:
        return []
    players = _g(state, "players", [])
    if player_idx >= len(players):
        return []
    p = players[player_idx]
    active_list = _g(p, "active", []) or []
    bench_list  = _g(p, "bench",  []) or []
    return [a for a in active_list if a is not None] + [b for b in bench_list if b is not None]


def _opp_idx(state) -> int:
    my = _g(state, "yourIndex", 0)
    return 1 - my


def _my_player(state):
    players = _g(state, "players", []) or []
    idx = _g(state, "yourIndex", 0)
    return players[idx] if idx < len(players) else {}


def _opp_player(state):
    players = _g(state, "players", []) or []
    idx = _opp_idx(state)
    return players[idx] if idx < len(players) else {}


def _hand_card_ids(state) -> list[int]:
    my = _my_player(state)
    hand = _g(my, "hand", []) or []
    return [_card_id(c) for c in hand]


def _prize_count(state, player_idx: int) -> int:
    players = _g(state, "players", []) or []
    if player_idx >= len(players):
        return 6
    return len(_g(players[player_idx], "prize", []) or [])


# ─── デッキ読み込み ──────────────────────────────────────────────

def read_deck() -> list[int]:
    for path in ["deck.csv", "/kaggle_simulations/agent/deck.csv",
                 os.path.join(os.path.dirname(__file__), "deck.csv")]:
        if os.path.exists(path):
            with open(path) as f:
                lines = f.read().strip().split("\n")
            return [int(lines[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv not found")


# ─── メインエントリ ──────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return read_deck()

    state   = obs_dict.get("current")
    options = _g(select, "option", []) or []
    context = _g(select, "context", CTX_MAIN)
    max_cnt = _g(select, "maxCount", 1) or 1
    min_cnt = _g(select, "minCount", 0) or 0

    if not options:
        return []

    handler = {
        CTX_MAIN:               _main_phase,
        CTX_SETUP_ACTIVE:       _setup_active,
        CTX_SETUP_BENCH:        _setup_bench,
        CTX_SWITCH:             _choose_switch,
        CTX_TO_ACTIVE:          _choose_switch,
        CTX_TO_BENCH:           _to_bench,
        CTX_DAMAGE_COUNTER_ANY: _damage_counter_any,
        CTX_DAMAGE_COUNTER:     _damage_counter_fixed,
        CTX_ATTACK:             _attack_select,
        CTX_EVOLVE:             _evolve_select,
        CTX_DISCARD:            _discard_select,
        CTX_DISCARD_ENERGY:     _discard_energy,
        CTX_TO_HAND:            _to_hand,
        CTX_EFFECT_TARGET:      _effect_target,
    }.get(context)

    if handler:
        result = handler(options, state, select)
        if result is not None:
            k = max(min_cnt, min(max_cnt, len(result)))
            return result[:k]

    # フォールバック: スコアリング
    return _fallback(options, state, select, max_cnt, min_cnt)


# ─── コンテキスト別ハンドラ ──────────────────────────────────────

def _main_phase(options, state, select) -> list[int]:
    """メインフェーズ: 行動優先順位でスコアリング"""
    my    = _my_player(state)
    opp   = _opp_player(state)
    opp_active_list = _g(opp, "active", []) or []
    opp_active = opp_active_list[0] if opp_active_list else None
    opp_hp = _remaining_hp(opp_active)

    my_idx   = _g(state, "yourIndex", 0)
    opp_idx  = _opp_idx(state)
    my_prizes  = _prize_count(state, my_idx)
    opp_prizes = _prize_count(state, opp_idx)
    my_bench   = _g(my, "bench", []) or []
    energy_attached = _g(state, "energyAttached", False)
    supporter_played = _g(state, "supporterPlayed", False)
    retreated = _g(state, "retreated", False)

    # 相手のベンチ全体の状況
    opp_bench = _g(opp, "bench", []) or []
    opp_bench_hp = sorted([_remaining_hp(p) for p in opp_bench])

    # 手札のカードID
    hand_ids = _hand_card_ids(state)

    scored = []
    for i, opt in enumerate(options):
        score = _score_main(
            opt, i, opp_hp, opp_bench_hp, my_prizes, opp_prizes,
            len(my_bench), energy_attached, supporter_played, retreated,
            hand_ids, state
        )
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [scored[0][1]]


def _score_main(opt, idx, opp_hp, opp_bench_hp, my_prizes, opp_prizes,
                bench_size, energy_attached, supporter_played, retreated,
                hand_ids, state) -> float:
    ot = _opt_type(opt)
    card_id = _g(opt, "cardId", 0) or 0

    # ── ATTACK ──────────────────────────────────────────
    if ot == OPT_ATTACK:
        score = 100.0
        # Jet Headbutt (70dmg) vs Phantom Dive (200dmg+bench)
        # 攻撃インデックスが高いほど = Phantom Dive 優先
        score += idx * 5
        # 終盤優先
        if opp_prizes <= 2:
            score += 60.0
        return score + random.uniform(0, 1)

    # ── ABILITY ─────────────────────────────────────────
    if ot == OPT_ABILITY:
        score = 80.0
        # Cursed Blast: 相手にKO圏内のポケモンがいれば最優先
        ability_card_id = _g(opt, "cardId", 0)
        if ability_card_id == DUSKNOIR:
            # 130点でKOできる相手がいるか
            targets = [hp for hp in [opp_hp] + opp_bench_hp if 0 < hp <= DUSKNOIR_CURSED_DMG]
            if targets:
                score += 200.0
            else:
                # 将来的に有効でなければ温存
                score -= 30.0
        elif ability_card_id == DUSCLOPS:
            targets = [hp for hp in [opp_hp] + opp_bench_hp if 0 < hp <= DUSCLOPS_CURSED_DMG]
            if targets:
                score += 150.0
            else:
                score -= 20.0
        return score + random.uniform(0, 1)

    # ── EVOLVE ──────────────────────────────────────────
    if ot == OPT_EVOLVE:
        evolve_to = _g(opt, "cardId", 0)
        priority = {
            DRAGAPULT_EX: 75.0,
            DUSKNOIR:     70.0,
            DRAKLOAK:     55.0,
            DUSCLOPS:     50.0,
        }
        return priority.get(evolve_to, 60.0) + random.uniform(0, 1)

    # ── PLAY (手からカードをプレイ) ──────────────────────
    if ot == OPT_PLAY:
        play_id = card_id
        score = _play_card_score(play_id, my_prizes, opp_prizes, opp_hp,
                                 opp_bench_hp, bench_size, supporter_played,
                                 energy_attached)
        return score + random.uniform(0, 1)

    # ── ATTACH ──────────────────────────────────────────
    if ot == OPT_ATTACH:
        return (45.0 if not energy_attached else -5.0) + random.uniform(0, 1)

    # ── RETREAT ─────────────────────────────────────────
    if ot == OPT_RETREAT:
        if retreated:
            return -100.0
        cost = _g(opt, "count", 2) or 2
        return (30.0 - cost * 10) + random.uniform(0, 1)

    # ── END ─────────────────────────────────────────────
    if ot == OPT_END:
        return -30.0 + random.uniform(0, 1)

    return random.uniform(0, 2)


def _play_card_score(card_id, my_prizes, opp_prizes, opp_hp, opp_bench_hp,
                     bench_size, supporter_played, energy_attached) -> float:
    """手からカードをプレイする際のスコア"""

    # ベーシックポケモンをベンチに出す: ベンチが空なら最優先
    if card_id in (DREEPY, DUSKULL, FEZANDIPITI_EX, BUDEW):
        base = 40.0
        if bench_size == 0:
            base += 120.0  # ベンチ空 = 場切れ負け回避
        elif bench_size <= 2:
            base += 30.0
        return base

    # サポーター: 1ターン1枚制限
    if card_id in (HILDA, CRISPIN, BOSSS_ORDERS) and supporter_played:
        return -50.0

    if card_id == UNFAIR_STAMP:
        # KO後条件はエンジン側で判定。使用できる状況なら最優先
        return 90.0

    if card_id == BOSSS_ORDERS:
        ko_targets = [hp for hp in opp_bench_hp if 0 < hp <= 130]
        return (85.0 + len(ko_targets) * 20) if ko_targets else 50.0

    if card_id == HILDA:
        return 70.0

    if card_id == CRISPIN:
        return 65.0 if not energy_attached else 50.0

    if card_id == JUDGE:
        # 相手の手が多い(序中盤)ほど効果大
        return 60.0

    if card_id == ULTRA_BALL:
        return 60.0

    if card_id == RARE_CANDY:
        return 55.0

    if card_id == BUDDY_BUDDY_POFFIN:
        return 50.0 if bench_size <= 3 else 20.0

    if card_id == DUSK_BALL:
        return 48.0

    if card_id == BUG_CATCHING_SET:
        return 46.0

    if card_id == POKEGEAR:
        return 44.0 if not supporter_played else 20.0

    return 20.0


def _setup_active(options, state, select) -> list[int]:
    """初期配置: Active選択 → Dreepy優先"""
    priority = [DREEPY, DUSKULL, DUSCLOPS, FEZANDIPITI_EX, BUDEW, DRAGAPULT_EX]
    for target in priority:
        for i, opt in enumerate(options):
            if _g(opt, "cardId", 0) == target:
                return [i]
    return [0]


def _setup_bench(options, state, select) -> list[int]:
    """初期配置: ベンチ選択 → 複数枚 (Dreepy, Duskull 優先)"""
    priority = [DREEPY, DUSKULL, FEZANDIPITI_EX, BUDEW, DRAKLOAK, DUSCLOPS]
    order = []
    used = set()
    for target in priority:
        for i, opt in enumerate(options):
            if i not in used and _g(opt, "cardId", 0) == target:
                order.append(i)
                used.add(i)
    # 残りも追加
    for i in range(len(options)):
        if i not in used:
            order.append(i)

    max_cnt = _g(select, "maxCount", 5) or 5
    return order[:max_cnt]


def _choose_switch(options, state, select) -> list[int]:
    """KO後のActive交替: 安いポケモンから出す (exを温存)"""
    priority = [DREEPY, DUSKULL, DUSCLOPS, BUDEW, FEZANDIPITI_EX, DUSKNOIR, DRAGAPULT_EX]
    for target in priority:
        for i, opt in enumerate(options):
            if _g(opt, "cardId", 0) == target:
                return [i]
    return [0]


def _to_bench(options, state, select) -> list[int]:
    """ベンチに出すポケモンを選択 (Buddy-Buddy Poffin等)"""
    priority = [DREEPY, DUSKULL, BUDEW, FEZANDIPITI_EX]
    max_cnt = _g(select, "maxCount", 2) or 2
    selected = []
    used = set()
    for target in priority:
        if len(selected) >= max_cnt:
            break
        for i, opt in enumerate(options):
            if i not in used and _g(opt, "cardId", 0) == target:
                selected.append(i)
                used.add(i)
    if not selected:
        selected = list(range(min(max_cnt, len(options))))
    return selected[:max_cnt]


def _damage_counter_any(options, state, select) -> list[int]:
    """
    Phantom Dive のベンチダメカン最適配置。
    remainDamageCounter = 残りカウンター数。
    1回の呼び出しで1体選択 → エンジンが繰り返し呼ぶ。

    方針:
    1. KO圏内(残りHP ≤ 残りカウンター×10)に入れる相手がいれば集中
    2. なければ残りHPが最も少ない相手ベンチに分散
    3. 相手Activeは200点で既にダメージ入るので後回し
    """
    remain = _g(select, "remainDamageCounter", 6) or 6
    damage_available = remain * 10

    # option の各ポケモンの残りHPを取得
    # CARD option: area/index でどのポケモンか特定できる
    # シンプルに: hp情報をoptから取れるか試みる
    # → optにhpがない場合は state から引く
    opp_idx = _opp_idx(state)
    opp_all = _all_field_pokemon(state, opp_idx)

    # serial → pokemon マッピング
    serial_to_pokemon = {_g(p, "serial", -1): p for p in opp_all}

    best_idx = 0
    best_score = -999.0

    for i, opt in enumerate(options):
        serial = _g(opt, "serial", -1)
        area   = _g(opt, "area", 5)     # 4=ACTIVE, 5=BENCH
        is_bench = (area == 5)

        # state から HP を引く
        pokemon = serial_to_pokemon.get(serial)
        if pokemon:
            hp = _remaining_hp(pokemon)
        else:
            hp = 100  # 不明なら適当な値

        # スコアリング
        score = 0.0

        # KOできるなら+1000
        if hp <= damage_available:
            score += 1000.0 - hp  # より少ないHPで取れる方が確実

        # ベンチ相手に撒く方が有益(Activeは攻撃で取れる)
        if is_bench:
            score += 10.0

        # 残りHPが少ない = 次のターンに繋がりやすい
        score += (300 - hp) * 0.5

        if score > best_score:
            best_score = score
            best_idx = i

    return [best_idx]


def _damage_counter_fixed(options, state, select) -> list[int]:
    """ダメカン配置(固定対象: DAMAGE_COUNTER)"""
    # 選択肢が1つの場合が多い
    return [0]


def _attack_select(options, state, select) -> list[int]:
    """攻撃選択: Phantom Dive 優先 (インデックスが大きい方)"""
    if not options:
        return [0]
    # Phantom Dive は Jet Headbutt より後ろに定義 → 最後のATTACKオプション
    attack_opts = [(i, opt) for i, opt in enumerate(options)]
    # 最後の選択肢 = より強い攻撃とみなす
    return [attack_opts[-1][0]]


def _evolve_select(options, state, select) -> list[int]:
    """進化選択: 上位進化を優先"""
    priority_to = [DRAGAPULT_EX, DUSKNOIR, DRAKLOAK, DUSCLOPS]
    for target in priority_to:
        for i, opt in enumerate(options):
            if _g(opt, "cardId", 0) == target:
                return [i]
    return [0]


def _discard_select(options, state, select) -> list[int]:
    """
    Ultra Ball等のコスト捨て: 2枚捨てる。
    優先して捨てるもの:
    - 余分なエネルギー
    - 手に複数ある Supporter
    - 使用条件が揃っていないItem
    """
    max_cnt = _g(select, "maxCount", 2) or 2

    # 優先捨て対象のカードIDリスト (価値が低いもの順)
    discard_priority = [
        PSYCHIC_ENERGY,  # エネルギーは山に多い
        FIRE_ENERGY,
        JUDGE,           # 使いやすい時に使えばよい
        CRISPIN,         # エネルギーが手にあれば重複OK
        POKEGEAR,
        DUSK_BALL,
        BUG_CATCHING_SET,
    ]
    selected = []
    used = set()
    for target_id in discard_priority:
        if len(selected) >= max_cnt:
            break
        for i, opt in enumerate(options):
            if i not in used and _g(opt, "cardId", 0) == target_id:
                selected.append(i)
                used.add(i)
                break

    # 足りなければ残りから適当に
    for i in range(len(options)):
        if len(selected) >= max_cnt:
            break
        if i not in used:
            selected.append(i)

    return selected[:max_cnt]


def _discard_energy(options, state, select) -> list[int]:
    """エネルギー捨て選択: 多い方のエネルギーを捨てる"""
    # Fire/Psychicの枚数を数えてより多い方を捨てる
    fire_opts    = [(i, o) for i, o in enumerate(options) if _g(o, "cardId", 0) == FIRE_ENERGY]
    psychic_opts = [(i, o) for i, o in enumerate(options) if _g(o, "cardId", 0) == PSYCHIC_ENERGY]
    if fire_opts:
        return [fire_opts[0][0]]
    if psychic_opts:
        return [psychic_opts[0][0]]
    return [0]


def _to_hand(options, state, select) -> list[int]:
    """
    山からカードを手に加える (Hilda, Ultra Ball 等)。
    優先: Dragapult ex > Dusknoir > Drakloak > Dusclops > Dreepy > Duskull > その他
    """
    max_cnt = _g(select, "maxCount", 1) or 1
    priority = [
        DRAGAPULT_EX, DUSKNOIR, DRAKLOAK, DUSCLOPS,
        DREEPY, DUSKULL, FEZANDIPITI_EX, RARE_CANDY,
        ULTRA_BALL, BUDDY_BUDDY_POFFIN, DUSK_BALL,
        BUG_CATCHING_SET, POKEGEAR,
        HILDA, CRISPIN, BOSSS_ORDERS, UNFAIR_STAMP, JUDGE,
        PSYCHIC_ENERGY, FIRE_ENERGY, BUDEW,
    ]
    selected = []
    used = set()
    for target_id in priority:
        if len(selected) >= max_cnt:
            break
        for i, opt in enumerate(options):
            if i not in used and _g(opt, "cardId", 0) == target_id:
                selected.append(i)
                used.add(i)
    # 足りなければ先頭から
    for i in range(len(options)):
        if len(selected) >= max_cnt:
            break
        if i not in used:
            selected.append(i)
    return selected[:max_cnt]


def _effect_target(options, state, select) -> list[int]:
    """効果対象選択: Boss's Orders等 → 残りHPが最も少ない相手ベンチ"""
    opp_idx = _opp_idx(state)
    opp_all = _all_field_pokemon(state, opp_idx)
    serial_to_pokemon = {_g(p, "serial", -1): p for p in opp_all}

    best_idx = 0
    best_hp = 9999
    for i, opt in enumerate(options):
        serial = _g(opt, "serial", -1)
        pk = serial_to_pokemon.get(serial)
        hp = _remaining_hp(pk) if pk else 9999
        if hp < best_hp:
            best_hp = hp
            best_idx = i
    return [best_idx]


def _fallback(options, state, select, max_cnt, min_cnt) -> list[int]:
    """YES/NO + フォールバック"""
    for i, opt in enumerate(options):
        if _opt_type(opt) == OPT_YES:
            return [i]
    k = max(min_cnt, min(max_cnt, len(options)))
    return list(range(k))
