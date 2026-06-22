"""
PTCG AI Battle Challenge - Iono's Bellibolt ex ヒューリスティック対戦相手

Kaggle Notebook "A Sample Rule-Based Agent Iono's Deck" (kiyotah, 運営公式サンプル)
(https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck)
の移植。Bellibolt exの特性でエネルギーを大量に貼ってVoltorbのVoltaic Chainで
高火力を出すLightningデッキ。主力アタッカーのVoltorbは非ex(倒されてもプライズ1枚)
なのが特徴で、ex/megaEx偏重の他の対戦相手(Lucario/Dragapult/Abomasnow)とは
リスク構造が異なるため、対戦相手プールの多様性確保用に追加する。
"""

import os
from collections import defaultdict

from cg.api import AreaType, Card, CardType, Observation, OptionType, Pokemon, SelectContext, all_card_data, to_observation_class


IONO_VOLTORB = 265
IONO_TADBULB = 268
IONO_BELLIBOLT_EX = 269
IONO_WATTREL = 270
IONO_KILOWATTREL = 271
BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
MAX_ROD = 1110
ENERGY_RETRIEVAL = 1118
ULTRA_BALL = 1121
POKE_PAD = 1152
LILLIE_DETERMINATION = 1227
CANARI = 1233
LEVINCIA = 1254
BASIC_LIGHTNING_ENERGY = 4

_all_card = all_card_data()
card_table = {c.cardId: c for c in _all_card}


def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "deck_iono.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck_iono.csv"
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
    return [int(x) for x in lines]


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def _legal_fallback(select) -> list[int]:
    try:
        n = len(select.option)
        k = min(max(0, select.minCount), n)
        return list(range(k))
    except Exception:
        return []


_can_attack = False


def _agent_impl(obs: Observation) -> list[int]:
    global _can_attack
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    op_prize = len(op_state.prize)

    field_counts: dict[int, int] = defaultdict(int)
    field_hand_counts: dict[int, int] = defaultdict(int)
    active_attacker = False
    bench_attacker = False

    energy_count = 0
    can_ability = False
    for p in my_state.active:
        if p is None:
            continue
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        energy_count += len(p.energies)
        if p.id == IONO_KILOWATTREL and len(p.energies) > 0:
            can_ability = True
        if p.id == IONO_VOLTORB and len(p.energies) >= 2:
            active_attacker = True
    for p in my_state.bench:
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        energy_count += len(p.energies)
        if p.id == IONO_KILOWATTREL and len(p.energies) > 0:
            can_ability = True
        if p.id == IONO_VOLTORB and len(p.energies) >= 2:
            bench_attacker = True

    field_pokemon1 = field_counts[IONO_TADBULB] + field_counts[IONO_BELLIBOLT_EX]
    field_pokemon2 = field_counts[IONO_WATTREL] + field_counts[IONO_KILOWATTREL]
    no_more_pokemon = len(my_state.bench) >= 5
    if field_counts[IONO_TADBULB] + field_counts[IONO_WATTREL] >= 1:
        no_more_pokemon = False

    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    hand_counts: dict[int, int] = defaultdict(int)
    hand_scores = []
    unused_hand_count = 0
    for c in my_state.hand:
        score = -10000
        if c.id == IONO_VOLTORB:
            score = 100
        elif c.id == IONO_BELLIBOLT_EX:
            if field_counts[c.id] <= 1:
                score = 120
        elif c.id == IONO_KILOWATTREL:
            if field_counts[c.id] <= 1:
                score = 140
        elif c.id == ULTRA_BALL:
            if not no_more_pokemon:
                score = 10
        elif c.id == NIGHT_STRETCHER:
            score = 50
        elif c.id == ENERGY_RETRIEVAL:
            score = 20
        elif c.id == MAX_ROD:
            score = 1000
        elif c.id == LILLIE_DETERMINATION:
            score = 150
        elif c.id == CANARI:
            score = 160
        elif c.id == LEVINCIA:
            if stadium_id != LEVINCIA:
                score = 30
        elif c.id == BASIC_LIGHTNING_ENERGY:
            score = -10
        score -= hand_counts[c.id] * 100
        hand_scores.append(score)
        if score < 0:
            unused_hand_count += 1
        hand_counts[c.id] += 1
        field_hand_counts[c.id] += 1

    discard_counts: dict[int, int] = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    if context == SelectContext.MAIN:
        _can_attack = any(o.type == OptionType.ATTACK for o in select.option)

    op_active_hp = 10000
    if len(op_state.active) >= 1 and op_state.active[0] is not None:
        op_active_hp = op_state.active[0].hp

    no_draw = my_state.deckCount <= 5

    scores = []
    id_counts: dict[int, int] = defaultdict(int)
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.ATTACH or context == SelectContext.ATTACH_FROM:
            if o.type == OptionType.ATTACH:
                p = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            else:
                p = get_card(obs, o.area, o.index, o.playerIndex)
            score = 40000
            if p.id == IONO_VOLTORB:
                if len(p.energies) >= 2:
                    if o.inPlayArea == AreaType.ACTIVE and not _can_attack:
                        score += 3000
                else:
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 5000
                    elif bench_attacker or active_attacker:
                        score += 100
                    else:
                        score += 1000
            elif p.id == IONO_TADBULB:
                score += 10 - len(p.energies)
            elif p.id == IONO_BELLIBOLT_EX:
                if len(p.energies) >= 4:
                    if o.inPlayArea == AreaType.ACTIVE and not _can_attack:
                        score += 500
                else:
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 800
                    elif bench_attacker or active_attacker:
                        score += 14 - len(p.energies)
                    else:
                        score += 100
            elif p.id == IONO_WATTREL:
                if len(p.energies) >= 1 or o.inPlayArea == AreaType.ACTIVE:
                    score += 10 - len(p.energies)
                else:
                    score += 6000
            elif p.id == IONO_KILOWATTREL:
                if len(p.energies) >= 1:
                    score += 11 - len(p.energies)
                else:
                    score += 8000
        elif o.type == OptionType.CARD:
            c = get_card(obs, o.area, o.index, o.playerIndex)
            if c is not None:
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    energy = 0
                    if isinstance(c, Pokemon):
                        energy = len(c.energies)
                        score -= c.hp
                        score -= energy * 100
                    if c.id == IONO_VOLTORB:
                        if 20 + energy_count * 20 >= op_active_hp:
                            score += 100000
                        else:
                            score += 1500
                        if energy >= 1:
                            score += 200
                            if energy >= 2:
                                score += 10000
                    elif c.id == IONO_BELLIBOLT_EX:
                        score += 1000
                        if energy >= 4:
                            score += 1000
                    elif c.id == IONO_TADBULB:
                        score += 10
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH):
                    if c.id == BASIC_LIGHTNING_ENERGY:
                        score += 1
                    elif c.id == IONO_VOLTORB:
                        if o.area == AreaType.DISCARD:
                            score += 100000
                        if field_counts[c.id] == 0:
                            score += 110
                        elif field_counts[c.id] == 1 and op_prize >= 2:
                            score += 5
                    elif c.id == IONO_TADBULB:
                        if field_pokemon1 == 0:
                            score += 200
                        elif field_pokemon1 == 1:
                            if op_prize >= 3 or (op_prize >= 2 and field_counts[IONO_BELLIBOLT_EX] == 0):
                                score += 20
                    elif c.id == IONO_BELLIBOLT_EX:
                        if field_hand_counts[c.id] == 0:
                            score += 250
                            if field_counts[IONO_TADBULB] > 0:
                                score += 300
                        elif field_hand_counts[c.id] == 1:
                            if op_prize >= 3:
                                score += 30
                                if field_counts[IONO_TADBULB] > 0:
                                    score += 30
                    elif c.id == IONO_WATTREL:
                        if field_pokemon2 == 0:
                            score += 320
                        elif field_pokemon2 == 1:
                            score += 15
                    elif c.id == IONO_KILOWATTREL:
                        if field_hand_counts[c.id] == 0:
                            score += 300
                            if field_counts[IONO_WATTREL] > 0:
                                score += 250
                        elif field_hand_counts[c.id] == 1:
                            score += 25
                            if field_counts[IONO_WATTREL] > 0:
                                score += 25

                    if c.id != BASIC_LIGHTNING_ENERGY:
                        if hand_counts[c.id] >= 2:
                            score -= 20000
                        elif hand_counts[c.id] >= 1:
                            score -= 2000
                        if id_counts[c.id] == 1:
                            score -= 1000
                        elif id_counts[c.id] >= 2:
                            score -= 10000
                    id_counts[c.id] += 1
                elif context == SelectContext.DISCARD:
                    if o.area == AreaType.HAND and o.playerIndex == my_index:
                        score = -hand_scores[o.index]
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(c.id)
            if data is not None and data.cardType == CardType.STADIUM:
                score = 85000 if (discard_counts[BASIC_LIGHTNING_ENERGY] >= 1 or can_ability) else -1
            elif data is not None and data.cardType == CardType.SUPPORTER:
                score = 25000
                if c.id == LILLIE_DETERMINATION:
                    score += 1000
                elif no_draw:
                    score = -1
                elif c.id == CANARI:
                    if no_more_pokemon:
                        score = -1
                    elif (
                        field_counts[IONO_VOLTORB] > 0
                        and field_counts[IONO_BELLIBOLT_EX] > 0
                        and field_counts[IONO_KILOWATTREL] > 0
                    ):
                        score += 100
                    else:
                        score += 2000
            elif data is not None and data.cardType == CardType.POKEMON:
                score = 100000
                if c.id == IONO_VOLTORB and field_counts[IONO_VOLTORB] >= 2:
                    score = -1
                elif c.id == IONO_TADBULB and field_pokemon1 >= 2:
                    score = -1
                elif c.id == IONO_WATTREL and field_pokemon2 >= 2:
                    if op_prize >= 2 or field_counts[IONO_VOLTORB] == 0 or field_counts[IONO_BELLIBOLT_EX] == 0:
                        score = -1
            else:
                if c.id == NIGHT_STRETCHER:
                    ok = (
                        discard_counts[IONO_VOLTORB] > 0
                        or (discard_counts[IONO_BELLIBOLT_EX] > 0 and field_counts[IONO_TADBULB] > 0)
                        or (discard_counts[IONO_KILOWATTREL] > 0 and field_counts[IONO_WATTREL] > 0)
                    )
                    score = 75000 if ok else -1
                elif c.id == ENERGY_RETRIEVAL:
                    score = 61000
                elif c.id == MAX_ROD:
                    score = 55000 if (state.turn >= 3 and discard_counts[BASIC_LIGHTNING_ENERGY] >= 2) else -1
                elif no_draw:
                    score = -1
                elif c.id == BUDDY_BUDDY_POFFIN:
                    score = 80000
                elif c.id == ULTRA_BALL:
                    if no_more_pokemon or state.turn <= 2:
                        score = -1
                    elif field_hand_counts[IONO_BELLIBOLT_EX] > 0 and field_hand_counts[IONO_KILOWATTREL] > 0:
                        score = 45000 if unused_hand_count >= 2 else -1
                    else:
                        score = 62000 if unused_hand_count >= 1 else -1
                elif c.id == POKE_PAD:
                    score = 79000
        elif o.type == OptionType.EVOLVE:
            score = 110000
        elif o.type == OptionType.ABILITY:
            score = -1
            c = get_card(obs, o.area, o.index, my_index)
            if c.id == IONO_BELLIBOLT_EX:
                score = 50000
            elif c.id == LEVINCIA:
                score = 8000
            elif not no_draw and c.id == IONO_KILOWATTREL:
                score = 30000
        elif o.type == OptionType.RETREAT:
            score = 10000 if (bench_attacker and not active_attacker) else -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    ranked = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return ranked[: select.maxCount]


def agent(obs_dict: dict) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return read_deck_csv() if obs_dict.get("select") is None else [0]

    if obs.select is None:
        return read_deck_csv()

    try:
        return _agent_impl(obs)
    except Exception:
        return _legal_fallback(obs.select)
