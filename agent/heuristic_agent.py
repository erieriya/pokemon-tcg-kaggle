"""
PTCG AI Battle Challenge - Heuristic Agent

cabt Engine API に基づいた正確なフィールド名を使用するヒューリスティクスエージェント。

観測構造:
  obs_dict["current"]["players"][i]:
    active: list[Pokemon|None]  (0 or 1 items)
    bench: list[Pokemon]
    hand: list[Card]|None  (自分のみ、相手はNone)
    prize: list[Card|None]
    deckCount: int
    handCount: int
    poisoned/burned/asleep/paralyzed/confused: bool

  obs_dict["select"]:
    type: SelectType
    context: SelectContext
    option: list[Option]
    minCount/maxCount: int
    remainDamageCounter/remainEnergyCost: int

  Option:
    type: OptionType (ATTACK=13, EVOLVE=9, ATTACH=8, PLAY=7, RETREAT=12, END=14, etc.)
    attackId: int (OptionType.ATTACK)
    cardId: int (カードの参照)
    serial: int
    inPlayArea/inPlayIndex: int

OptionType値:
  0=NUMBER, 1=YES, 2=NO, 3=CARD, 4=TOOL_CARD, 5=ENERGY_CARD,
  6=ENERGY, 7=PLAY, 8=ATTACH, 9=EVOLVE, 10=ABILITY,
  11=DISCARD, 12=RETREAT, 13=ATTACK, 14=END, 15=SKILL, 16=SPECIAL_CONDITION
"""

import random
from typing import Optional, Any


# OptionType 定数
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16


def _get_option_type(option) -> int:
    """Option の type フィールドを取得"""
    if isinstance(option, dict):
        return option.get("type", OPT_END)
    # dataclass instance の場合
    return getattr(option, "type", OPT_END)


def _get_field(obj, *keys, default=0):
    """dict または dataclass からフィールドを取得"""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            obj = getattr(obj, key, default)
    return obj


def agent(obs_dict: dict) -> list[int]:
    """メインエージェント関数"""
    select = obs_dict.get("select", {}) or {}
    options = _get_field(select, "option", default=[])
    max_count = _get_field(select, "maxCount", default=1)
    min_count = _get_field(select, "minCount", default=0)

    if not options:
        return []

    current = obs_dict.get("current")

    # YES/NO 選択は状況に応じてYES優先
    for i, opt in enumerate(options):
        opt_type = _get_option_type(opt)
        if opt_type == OPT_YES:
            return [i]

    # メインフェーズ: 優先順位でスコアリング
    scored = _score_all_options(options, current, select)
    scored.sort(key=lambda x: x[0], reverse=True)

    k = min(max_count, len(options))
    k = max(k, min_count)
    return [idx for _, idx in scored[:k]]


def _score_all_options(options: list, current, select) -> list[tuple[float, int]]:
    """全オプションにスコアをつける"""
    players = _get_field(current, "players", default=[]) if current else []
    my = players[0] if players else {}
    opp = players[1] if len(players) > 1 else {}

    my_active_list = _get_field(my, "active", default=[])
    my_active = my_active_list[0] if my_active_list else None
    opp_active_list = _get_field(opp, "active", default=[])
    opp_active = opp_active_list[0] if opp_active_list else None

    opp_hp_remaining = _get_remaining_hp(opp_active)
    my_prize_count = len(_get_field(my, "prize", default=[]))
    opp_prize_count = len(_get_field(opp, "prize", default=[]))
    my_bench_size = len(_get_field(my, "bench", default=[]))
    supporter_played = _get_field(current, "supporterPlayed", default=False) if current else False
    energy_attached = _get_field(current, "energyAttached", default=False) if current else False
    retreated = _get_field(current, "retreated", default=False) if current else False

    scored = []
    for i, opt in enumerate(options):
        score = _score_option(
            opt, i, opp_hp_remaining, my_prize_count, opp_prize_count,
            my_bench_size, supporter_played, energy_attached, retreated
        )
        scored.append((score, i))
    return scored


def _score_option(opt, idx: int, opp_hp: int, my_prizes: int, opp_prizes: int,
                  my_bench: int, supporter_played: bool, energy_attached: bool,
                  retreated: bool) -> float:
    opt_type = _get_option_type(opt)

    score_map = {
        OPT_ATTACK: 100.0,
        OPT_EVOLVE: 65.0,
        OPT_ABILITY: 60.0,
        OPT_ATTACH: 50.0 if not energy_attached else -10.0,
        OPT_PLAY: 40.0,
        OPT_SKILL: 35.0,
        OPT_TOOL_CARD: 30.0,
        OPT_ENERGY_CARD: 25.0,
        OPT_RETREAT: 10.0 if not retreated else -100.0,
        OPT_YES: 5.0,
        OPT_NO: 3.0,
        OPT_DISCARD: 2.0,
        OPT_END: -20.0,
        OPT_NUMBER: 0.0,
        OPT_CARD: 0.0,
        OPT_ENERGY: 0.0,
        OPT_SPECIAL_CONDITION: 0.0,
    }

    base = score_map.get(opt_type, 0.0)

    if opt_type == OPT_ATTACK:
        damage = _get_field(opt, "damage", default=0)
        # KOできるなら最優先
        if opp_hp > 0 and damage >= opp_hp:
            base += 300.0
        # 相手のサイドが少ない（終盤）ほど攻撃を優先
        if opp_prizes <= 2:
            base += 50.0
        base += min(damage, 300) / 10.0

    # ベンチが空なら場に出すことを最優先（場切れ防止）
    if opt_type == OPT_PLAY and my_bench == 0:
        base += 100.0

    # 逃げることは最後の手段
    if opt_type == OPT_RETREAT:
        retreat_cost = _get_field(opt, "count", default=1)
        if retreat_cost == 0:
            base += 30.0

    return base + random.uniform(0, 2.0)


def _get_remaining_hp(pokemon) -> int:
    """ポケモンの残りHPを取得"""
    if pokemon is None:
        return 999
    max_hp = _get_field(pokemon, "maxHp", default=100)
    hp = _get_field(pokemon, "hp", default=max_hp)
    return max(0, hp)


def _get_retreat_cost(pokemon) -> int:
    if pokemon is None:
        return 0
    return len(_get_field(pokemon, "energies", default=[]))
