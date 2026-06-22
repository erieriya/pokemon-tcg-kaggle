"""
PTCG AI Battle Challenge - Crustle wall ヒューリスティック対戦相手

Kaggle Notebook "Beating the Day-1 #1 Crustle Bot"
(https://www.kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot)
の移植。

Crustle(345)の特性「Mysterious Rock Inn」は相手のex Pokemonからの攻撃ダメージを
完全に無効化する。このコンペで初日のランキング上位を占めたとされる、最重要の
メタ対策の一つ。うちの学習デッキ(Dragapult ex)・対戦相手のLucario系も
ex/megaEx主体のアタッカーなので、Crustleウォールを経験せずに学習すると
このメタへの対応を一度も学べない懸念があるため、対戦相手プールに追加する。

エージェント自体は意図的にシンプル(ATTACH→EVOLVE→PLAY→ABILITY→ATTACK→RETREATの
優先順位だけ)。元notebook曰く「強さの理由はデッキ側にあり、パイロットは馬鹿でもいい」
というのがこのデッキの売り文句。
"""

import os

from cg.api import AreaType, Card, Observation, OptionType, Pokemon, SelectContext, to_observation_class


def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "deck_crustle.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck_crustle.csv"
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


def agent(obs_dict: dict) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return read_deck_csv() if obs_dict.get("select") is None else [0]

    if obs.select is None:
        return read_deck_csv()

    try:
        select = obs.select
        options = select.option
        context = select.context

        scores = []
        for o in options:
            score = 0

            if context == SelectContext.MAIN:
                if o.type == OptionType.ATTACH:
                    score = 1000
                    card = get_card(obs, o.area, o.index, obs.current.yourIndex)
                    if card is not None and card.id == 1159:  # Hero's Cape
                        score = 2100 if o.inPlayArea == AreaType.ACTIVE else 0
                elif o.type == OptionType.EVOLVE:
                    score = 800
                elif o.type == OptionType.PLAY:
                    score = 600
                    card = get_card(obs, AreaType.HAND, o.index, obs.current.yourIndex)
                    if card is not None:
                        active = obs.current.players[obs.current.yourIndex].active
                        pokemon = active[0] if active and active[0] is not None else None
                        if card.id == 1147:  # Jumbo Ice Cream: heal 80 if 3+ energy and damaged
                            score = (
                                2000
                                if pokemon is not None and pokemon.hp < pokemon.maxHp and len(pokemon.energies) >= 3
                                else 0
                            )
                        elif card.id == 1212:  # Cook: heal 70 if damaged
                            score = 1500 if pokemon is not None and pokemon.hp < pokemon.maxHp else 0
                        elif card.id == 1224:  # Cheren: draw 3
                            score = 1400
                        elif card.id == 1264:  # Battle Cage
                            score = 1300
                elif o.type == OptionType.ABILITY:
                    score = 400
                elif o.type == OptionType.ATTACK:
                    score = 100
                elif o.type == OptionType.RETREAT:
                    score = -1
            else:
                score = 2000
                if o.type == OptionType.CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    if card is not None:
                        if context in (SelectContext.EVOLVE, SelectContext.TO_BENCH):
                            score += 500
                        if isinstance(card, Pokemon):
                            if o.playerIndex != obs.current.yourIndex:
                                score += 500 if o.area == AreaType.ACTIVE else 100
                                score += len(card.energies) * 50
                            else:
                                score += card.hp
                elif o.type == OptionType.YES:
                    score += 100
                elif o.type == OptionType.NUMBER:
                    score += o.number

            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        output = []
        for i in ranked[: select.maxCount]:
            if scores[i] >= 0 or len(output) < select.minCount:
                output.append(i)
        return output if output else _legal_fallback(select)
    except Exception:
        return _legal_fallback(obs.select)
