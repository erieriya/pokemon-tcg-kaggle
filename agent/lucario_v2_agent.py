"""
PTCG AI Battle Challenge - Lucario v2 ヒューリスティック対戦相手

Kaggle Notebook "Pokemon TCG — Lucario v2 Strategy Baseline"
(https://www.kaggle.com/code/pilkwang/pokemon-tcg-lucario-v2-strategy-baseline)
の LucarioPolicy (tempoデッキバリアント) をPPO自己対戦の固定対戦相手として
移植したもの。v1 (lucario_v1_agent.py) より多くのSelectContext
(ベンチ配置/捨て札/ダメカン配置/回復対象/進化元先/マリガン等)を
カバーしている。

元notebookの前向き探索層・診断(_DIAG)・検証ゲートは提出前検証用の
ツールなのでこの移植では省略し、greedyなスコアリング方策のみを採用している
(元notebookもUSE_SEARCH_IN_AGENT=Falseがデフォルトで、greedy方策が主力)。
"""

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 8

_all_card = all_card_data()
card_table = {card.cardId: card for card in _all_card}


def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "deck_lucario_v2.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck_lucario_v2.csv"
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
    return [int(x) for x in lines]


class AttackPlan:
    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def _ctx_is(context, *names) -> bool:
    for name in names:
        value = getattr(SelectContext, name, None)
        if value is not None and context == value:
            return True
    return False


def _safe_get(seq, index):
    try:
        if seq is None or index is None or index < 0 or index >= len(seq):
            return None
        return seq[index]
    except Exception:
        return None


def _pokemon_max_hp(pokemon: Pokemon) -> int:
    max_hp = getattr(pokemon, "maxHp", None)
    if max_hp is not None:
        return max_hp
    data = card_table.get(getattr(pokemon, "id", None))
    return getattr(data, "hp", getattr(pokemon, "hp", 0))


def _damage_on(pokemon: Pokemon) -> int:
    return max(0, _pokemon_max_hp(pokemon) - getattr(pokemon, "hp", 0))


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return _safe_get(getattr(obs.select, "deck", None), index)
        if area == AreaType.HAND:
            return _safe_get(getattr(player, "hand", None), index)
        if area == AreaType.DISCARD:
            return _safe_get(getattr(player, "discard", None), index)
        if area == AreaType.ACTIVE:
            return _safe_get(getattr(player, "active", None), index)
        if area == AreaType.BENCH:
            return _safe_get(getattr(player, "bench", None), index)
        if area == AreaType.PRIZE:
            return _safe_get(getattr(player, "prize", None), index)
        if area == AreaType.STADIUM:
            return _safe_get(getattr(obs.current, "stadium", None), index)
        if area == AreaType.LOOKING:
            return _safe_get(getattr(obs.current, "looking", None), index)
        return None
    except Exception:
        return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return prize_count(pokemon) * 1000 + getattr(pokemon, "hp", 0) + _damage_on(pokemon) * 2
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id in {144, 322, 323, 337}:
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


def normalize_selection(ranked: list[int], scores: list[float], select) -> list[int]:
    n = len(select.option)
    minc = max(0, min(select.minCount, n))
    maxc = max(minc, min(select.maxCount, n))

    out: list[int] = []
    seen: set[int] = set()
    for i in ranked:
        if not (0 <= i < n) or i in seen:
            continue
        score = scores[i] if i < len(scores) else 0
        if score > 0 or len(out) < minc:
            out.append(i)
            seen.add(i)
        if len(out) >= maxc:
            break

    for i in range(n):
        if len(out) >= minc:
            break
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _legal_fallback(select) -> list[int]:
    try:
        n = len(select.option)
        k = min(max(0, select.minCount), n)
        return list(range(k))
    except Exception:
        return []


def _legal_fallback_from_dict(obs_dict: dict) -> list[int]:
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        minc = sel.get("minCount", 0)
        n = len(opts)
        k = min(max(0, minc), n)
        return list(range(k))
    except Exception:
        return []


class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def rank(self) -> tuple[list[int], list[float]]:
        if not self.select.option or self.select.maxCount == 0:
            return [], []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked, scores

    def choose(self) -> list[int]:
        ranked, scores = self.rank()
        selection = normalize_selection(ranked, scores, self.select)
        self.remember_chosen_side_effects(selection)
        return selection

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is None:
                    continue
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is None:
                    continue
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self):
        return self.me.active + self.me.bench

    def _opponent_board(self):
        return self.opponent.active + self.opponent.bench

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int):
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self.my_prizes_left in {2, 3}:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    op_data = card_table.get(op_pokemon.id)
                    if op_data is None:
                        continue
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            score += 100 if energy_count < 3 else 0
            score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return self._score_setup_bench(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.TO_BENCH:
            return self._score_to_bench(card)
        if _ctx_is(self.context, "TO_FIELD"):
            return self._score_to_field(option, card)
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}:
            return self._score_damage_counter(option, card)
        if _ctx_is(self.context, "DAMAGE", "EFFECT_TARGET"):
            return self._score_effect_target(option, card)
        if _ctx_is(self.context, "HEAL", "REMOVE_DAMAGE_COUNTER"):
            return self._score_heal_target(option, card)
        if _ctx_is(self.context, "EVOLVES_FROM", "EVOLVES_TO"):
            return self._score_evolution_context(option, card)
        if _ctx_is(self.context, "MULLIGAN"):
            return self._score_mulligan(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8 if self.my_prizes_left in {2, 3} else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_setup_active(self, card) -> int:
        if card is None:
            return 0
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_setup_bench(self, card) -> int:
        if card is None:
            return 0
        if card.id == C.RIOLU:
            return 120 - 25 * self.field_counts[C.RIOLU]
        if card.id == C.SOLROCK:
            return 90 if self.field_counts[C.SOLROCK] == 0 else -1
        if card.id == C.LUNATONE:
            return 80 if self.field_counts[C.LUNATONE] == 0 else -1
        if card.id == C.MAKUHITA:
            return 65 if self.field_counts[C.MAKUHITA] == 0 else 10
        return 0

    def _score_to_bench(self, card) -> float:
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None or data.cardType != CardType.POKEMON:
            return 0
        return self._score_setup_bench(card)

    def _score_discard(self, card) -> float:
        if card is None:
            return 0
        cid = card.id
        if cid == C.BASIC_FIGHTING_ENERGY:
            score = 45 if self.hand_counts[cid] >= 2 else 5
            if plan.needs_energy and not self.state.energyAttached:
                score -= 200
            return score
        if self.hand_counts[cid] >= 2:
            return 70
        if cid in {C.LUNATONE, C.SOLROCK} and self.field_counts[cid] >= 1:
            return 55
        if cid == C.GRAVITY_MOUNTAIN and self.stadium_id == C.GRAVITY_MOUNTAIN:
            return 50
        if cid in {C.CARMINE, C.LILLIE_DETERMINATION} and self.state.supporterPlayed:
            return 30
        if cid == C.MEGA_LUCARIO_EX and self.field_counts[C.RIOLU] == 0:
            return -80
        if cid == C.HARIYAMA and self.field_counts[C.MAKUHITA] == 0:
            return -50
        if cid in {C.RIOLU, C.MAKUHITA, C.BOSS_ORDERS, C.HERO_CAPE}:
            return -40
        return 0

    def _is_opponent_option(self, option) -> bool:
        return getattr(option, "playerIndex", self.my_index) == self.op_index

    def _score_damage_counter(self, option, card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if self._is_opponent_option(option):
            return 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0) + _damage_on(card) * 5
        return -target_score(card)

    def _score_effect_target(self, option, card) -> float:
        if not isinstance(card, Pokemon):
            return self._score_to_hand(card)
        if self._is_opponent_option(option):
            return 2000 + target_score(card) + _damage_on(card) * 8
        score = 300 + len(card.energies) * 50 + len(card.tools) * 40 + _damage_on(card) * 8
        if card.id == C.MEGA_LUCARIO_EX:
            score += 250
        elif card.id in {C.RIOLU, C.HARIYAMA}:
            score += 120
        elif card.id in {C.SOLROCK, C.LUNATONE, C.MAKUHITA}:
            score += 70
        return score

    def _score_heal_target(self, option, card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        damage = _damage_on(card)
        if self._is_opponent_option(option):
            return -1000 - damage
        score = damage * 20 + len(card.energies) * 30 + len(card.tools) * 25
        if card.id == C.MEGA_LUCARIO_EX:
            score += 300
        elif card.id in {C.RIOLU, C.HARIYAMA}:
            score += 120
        return score if damage > 0 else max(0, score // 10)

    def _score_evolution_context(self, option, card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if self._is_opponent_option(option):
            return target_score(card)
        if card.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            return 900 + len(card.energies) * 30
        if card.id in {C.MAKUHITA, C.HARIYAMA}:
            return 650 + len(card.energies) * 30
        return 100 + len(card.energies) * 20

    def _score_to_field(self, option, card) -> float:
        if isinstance(card, Pokemon):
            return self._score_to_bench(card) + self._score_setup_active(card)
        return self._score_to_hand(card)

    def _score_mulligan(self, card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        is_basic = (
            getattr(data, "basic", False)
            or (
                data.cardType == CardType.POKEMON
                and not getattr(data, "stage1", False)
                and not getattr(data, "stage2", False)
                and not getattr(data, "megaEx", False)
            )
        )
        if not is_basic:
            return 0
        return self._score_setup_active(card) + self._score_setup_bench(card)

    def _score_to_hand(self, card) -> float:
        if card is None:
            return 0
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.MAKUHITA:
            score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card) -> float:
        score = 20000
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1
        return score

    def _score_play_trainer(self, card) -> float:
        if card.id == C.SWITCH:
            if plan.attacker > 0 and plan.remain_hp <= 0:
                return 14000
            return 6500 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            if plan.target >= 1 and plan.remain_hp <= 0:
                return 15000
            return 4200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            return -1 if self._low_deck() else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and card_table.get(pokemon.id) is not None and card_table[pokemon.id].stage2
            for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            score = 7000
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if getattr(pokemon, "appearThisTurn", False):
            return -1
        if pokemon.id == C.MAKUHITA and plan.target == 0:
            return -1
        evolve_card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        score = 9000 + len(pokemon.energies)
        if evolve_card is not None:
            if evolve_card.id == C.MEGA_LUCARIO_EX and pokemon.id == C.RIOLU:
                score += 350
            elif evolve_card.id == C.HARIYAMA and pokemon.id == C.MAKUHITA:
                score += 180
        return score

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        return 30000

    def remember_chosen_side_effects(self, selection: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN:
            return
        for idx in selection:
            if not (0 <= idx < len(self.select.option)):
                continue
            option = self.select.option[idx]
            if option.type != OptionType.ABILITY:
                continue
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.LUNATONE:
                ability_used = True


def agent(obs_dict: dict) -> list[int]:
    global pre_turn, ability_used, plan

    try:
        select_is_none = isinstance(obs_dict, dict) and obs_dict.get("select") is None
    except Exception:
        select_is_none = False
    if select_is_none:
        return read_deck_csv()

    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()

        if obs.current is not None and pre_turn != obs.current.turn:
            pre_turn = obs.current.turn
            ability_used = False
            plan = AttackPlan()

        try:
            policy = LucarioPolicy(obs)
            ranked, scores = policy.rank()
            selection = normalize_selection(ranked, scores, obs.select)
            policy.remember_chosen_side_effects(selection)
            return selection
        except Exception:
            return _legal_fallback(obs.select)
    except Exception:
        return _legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})
