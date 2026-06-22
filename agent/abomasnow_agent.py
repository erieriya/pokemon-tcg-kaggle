"""
PTCG AI Battle Challenge - Mega Abomasnow ex ヒューリスティック対戦相手

Kaggle Notebook "A Sample Rule-Based Agent Mega Abomasnow ex Deck" (kiyotah, 運営公式サンプル)
(https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck)
の移植。Water単色・34枚の基本エネルギーを採用したシンプルな殴り合いデッキで、
research.mdの4アーキタイプの一つ(対戦相手プールの多様性確保用)。
"""

import os
from collections import defaultdict

from cg.api import AreaType, Card, Observation, OptionType, Pokemon, SelectContext, to_observation_class


KYOGRE = 721
SNOVER = 722
MEGA_ABOMASNOW_EX = 723
ULTRA_BALL = 1121
PRECIOUS_TROLLEY = 1126
CARMINE = 1192
LILLIE_DETERMINATION = 1227
SURFING_BEACH = 1262
BASIC_WATER_ENERGY = 3


def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "deck_abomasnow.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck_abomasnow.csv"
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


def _agent_impl(obs: Observation) -> list[int]:
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]

    field_counts: dict[int, int] = defaultdict(int)
    hand_counts: dict[int, int] = defaultdict(int)
    discard_counts: dict[int, int] = defaultdict(int)

    bench_attacker_index0 = -1  # Mega Abomasnow ex ready on bench
    bench_attacker_index1 = -1  # Kyogre ready on bench
    for i, card in enumerate(my_state.bench):
        field_counts[card.id] += 1
        if card.id == MEGA_ABOMASNOW_EX and len(card.energies) >= 2:
            bench_attacker_index0 = i
        elif card.id == KYOGRE and len(card.energies) >= 1:
            bench_attacker_index1 = i

    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    op_active_hp = 0
    for card in state.players[1 - my_index].active:
        if card is None:
            continue
        op_active_hp = card.hp

    prefer_ky = op_active_hp <= 20 * discard_counts[BASIC_WATER_ENERGY]
    switch_index = -1
    for card in my_state.active:
        if card is None:
            continue
        field_counts[card.id] += 1
        if card.id == MEGA_ABOMASNOW_EX and len(card.energies) >= 2:
            if prefer_ky and bench_attacker_index1 >= 0:
                switch_index = bench_attacker_index1
        elif card.id == KYOGRE and len(card.energies) >= 1:
            if not prefer_ky and bench_attacker_index0 >= 0:
                switch_index = bench_attacker_index0
        elif bench_attacker_index0 >= 0:
            switch_index = bench_attacker_index0

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    score += energy_count * 2
                    if o.index == switch_index:
                        score += 100
                    if card.id == MEGA_ABOMASNOW_EX:
                        score += 20
                    elif card.id == KYOGRE:
                        score += 10
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
                    if card.id == SNOVER:
                        if field_counts[card.id] >= 1:
                            score += 5
                        elif field_counts[MEGA_ABOMASNOW_EX] >= 1:
                            score += 15
                        else:
                            score += 30
                    elif card.id == MEGA_ABOMASNOW_EX:
                        if field_counts[SNOVER] >= 1 and field_counts[card.id] + hand_counts[card.id] == 0:
                            score += 100
                        else:
                            score += 10
                    elif card.id == KYOGRE:
                        score += 1 if field_counts[card.id] >= 1 else 20
                elif context == SelectContext.DISCARD:
                    if card.id == BASIC_WATER_ENERGY:
                        score += 100
                    elif card.id == MEGA_ABOMASNOW_EX:
                        score += 10
                    elif card.id == CARMINE:
                        if hand_counts[LILLIE_DETERMINATION] >= 1:
                            score += 30
                    elif card.id == LILLIE_DETERMINATION:
                        score -= 20
                    if hand_counts[card.id] >= 2:
                        score += 500
                    hand_counts[card.id] -= 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            score = 10000
            if card.id == ULTRA_BALL:
                no_aboma_line = (
                    field_counts[MEGA_ABOMASNOW_EX] + hand_counts[MEGA_ABOMASNOW_EX] == 0
                    or field_counts[MEGA_ABOMASNOW_EX] + field_counts[SNOVER] == 0
                    or field_counts[KYOGRE] == 0
                )
                if hand_counts[BASIC_WATER_ENERGY] >= 3 or (my_state.handCount >= 4 and no_aboma_line):
                    score = 4000
                else:
                    score = -1
            elif card.id == CARMINE:
                score = -1 if (field_counts[SNOVER] >= 1 and hand_counts[MEGA_ABOMASNOW_EX] >= 1) else 3000
            elif card.id == LILLIE_DETERMINATION:
                bad = field_counts[SNOVER] >= 1 and field_counts[MEGA_ABOMASNOW_EX] == 0 and hand_counts[MEGA_ABOMASNOW_EX] >= 1
                score = -1 if bad else 3100
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 5000
            energy_count = len(pokemon.energies)
            if energy_count == 0 and o.inPlayArea == AreaType.BENCH:
                score += 1
            if pokemon.id == SNOVER:
                score += 1
                if energy_count == 1:
                    score -= 100
                elif energy_count >= 2:
                    score -= 400
                if bench_attacker_index0 >= 0:
                    score -= 300
            elif pokemon.id == MEGA_ABOMASNOW_EX:
                score += 10
                if energy_count == 1:
                    score += 30
                elif energy_count >= 2:
                    score -= 300
                if bench_attacker_index0 >= 0:
                    score -= 200
            elif pokemon.id == KYOGRE:
                score += 5
                if len(pokemon.energies) >= 1:
                    score -= 200
                if bench_attacker_index1 >= 0:
                    score -= 200
            if o.inPlayArea == AreaType.ACTIVE:
                if bench_attacker_index0 >= 0 and bench_attacker_index1 >= 0 and energy_count <= 2:
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 10000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            score = 2000 if (card.id == SURFING_BEACH and switch_index >= 0) else -1
        elif o.type == OptionType.RETREAT:
            score = 1500 if switch_index >= 0 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == 1042:  # Riptide
                score += discard_counts[BASIC_WATER_ENERGY] * 20 - 90
            elif o.attackId == 1046:  # Hammer-lanche
                score += -100 if op_active_hp <= 200 else 100

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
