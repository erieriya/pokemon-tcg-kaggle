import os
import random

from cg.api import AreaType, Observation, to_observation_class

# card IDs
DREEPY = 119; DRAKLOAK = 120; DRAGAPULT_EX = 121
DUSKULL = 131; DUSCLOPS = 132; DUSKNOIR = 133
FEZANDIPITI_EX = 140; BUDEW = 235
MEOWTH_EX = 1071; MUNKIDORI = 112; SHAYMIN = 343
ULTRA_BALL = 1121; RARE_CANDY = 1079; BUDDY_BUDDY_POFFIN = 1086
POKE_PAD = 1152; NIGHT_STRETCHER = 1097
UNFAIR_STAMP = 1080
CRISPIN = 1198; BOSSS_ORDERS = 1182; JUDGE = 1213
LILLIES_DETERMINATION = 1227; TEAM_ROCKETS_PETREL = 1219
TEAM_ROCKETS_WATCHTOWER = 1256; RISKY_RUINS = 1260
FIRE_ENERGY = 2; PSYCHIC_ENERGY = 5; DARKNESS_ENERGY = 7
DUSKNOIR_DMG = 130; DUSCLOPS_DMG = 50
EX_DAMAGE_IMMUNE_IDS = {345}  # Crustle「Mysterious Rock Inn」: 相手のexポケモンの攻撃ダメージを完全に防ぐ

# option types
OPT_YES = 1; OPT_NO = 2; OPT_PLAY = 7; OPT_ATTACH = 8
OPT_EVOLVE = 9; OPT_ABILITY = 10; OPT_RETREAT = 12
OPT_ATTACK = 13; OPT_END = 14


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def _card_id(opt) -> int:
    return getattr(opt, "cardId", 0) or 0


def _opt_type(opt) -> int:
    return int(getattr(opt, "type", OPT_END)) if opt is not None else OPT_END


def _hp(pokemon) -> int:
    if pokemon is None:
        return 9999
    return max(0, getattr(pokemon, "hp", 100))


def _opp_hps(obs: Observation) -> list[int]:
    if obs.current is None:
        return []
    oi = 1 - obs.current.yourIndex
    opp = obs.current.players[oi]
    active = opp.active or []
    bench = opp.bench or []
    return [_hp(p) for p in active + bench if p is not None]


def _opp_active_hp(obs: Observation) -> int | None:
    if obs.current is None:
        return None
    oi = 1 - obs.current.yourIndex
    opp = obs.current.players[oi]
    active = opp.active or []
    if not active or active[0] is None:
        return None
    return _hp(active[0])


def _opp_active_ex_immune(obs: Observation) -> bool:
    if obs.current is None:
        return False
    oi = 1 - obs.current.yourIndex
    opp = obs.current.players[oi]
    active = opp.active or []
    if not active or active[0] is None:
        return False
    return getattr(active[0], "id", None) in EX_DAMAGE_IMMUNE_IDS


def _opp_bench_has_ex_immune(obs: Observation) -> bool:
    """相手のベンチにEX_DAMAGE_IMMUNE_IDSに含まれるポケモンがいればTrue。"""
    if obs.current is None:
        return False
    oi = 1 - obs.current.yourIndex
    opp = obs.current.players[oi]
    bench = opp.bench or []
    return any(
        getattr(pokemon, "id", None) in EX_DAMAGE_IMMUNE_IDS
        for pokemon in bench
        if pokemon is not None
    )


def _my_active_is_ex(obs: Observation, my_index: int) -> bool:
    if obs.current is None:
        return False
    active = obs.current.players[my_index].active or []
    if not active or active[0] is None:
        return False
    data = _card_data(getattr(active[0], "id", None))
    return bool(data and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))


_ATTACK_DB: dict[int, object] | None = None
_CARD_DB: dict[int, object] | None = None


def _card_data(card_id):
    """cardIdからCardDataを引く（all_card_data()を初回のみ読み込みキャッシュ）。"""
    global _CARD_DB
    if _CARD_DB is None:
        from cg.api import all_card_data
        _CARD_DB = {card.cardId: card for card in all_card_data()}
    return _CARD_DB.get(card_id)


def _enumerate_distributions(total: int, n: int):
    """totalを非負整数のn個組に分配する全パターンを列挙するジェネレータ。"""
    if n == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _enumerate_distributions(total - first, n - 1):
            yield (first,) + rest


def _plan_damage_distribution(candidates: list[dict], remain: int, own_pz: int) -> list[int]:
    """
    残りのダメカンを候補へ分配する全パターンを評価し、最高スコアの配分を返す。
    """
    if not candidates:
        return []

    best_distribution = None
    best_score = float("-inf")
    for distribution in _enumerate_distributions(remain, len(candidates)):
        score = 0.0
        for candidate, count in zip(candidates, distribution):
            damage = count * 10
            prize_value = candidate["prize_value"]
            if damage >= candidate["hp"]:
                score += prize_value * 1000.0
                if prize_value >= 2.0 and own_pz <= 2:
                    score -= 400.0
            else:
                score += damage * 0.5
        if score > best_score:
            best_score = score
            best_distribution = distribution

    return list(best_distribution)


def _get_card_or_pokemon(obs, area, index, player_index):
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return player.hand[index]
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def _remaining_cost(current_energies: list, extra_type: int | None, required: list[int]) -> int:
    """現在のエネルギー(+仮の1枚)でワザ要求を満たすまでの不足数を返す。"""
    pool = [int(energy) for energy in current_energies]
    if extra_type is not None:
        pool.append(int(extra_type))
    colorless_needed = 0
    missing = 0
    for energy in required:
        req = int(energy)
        if req == 0:
            colorless_needed += 1
        elif req in pool:
            pool.remove(req)
        else:
            missing += 1
    return missing + max(0, colorless_needed - len(pool))


def _attack_data(attack_id):
    global _ATTACK_DB
    if _ATTACK_DB is None:
        from cg.api import all_attack
        _ATTACK_DB = {attack.attackId: attack for attack in all_attack()}
    return _ATTACK_DB.get(attack_id)


def _attack_damage(attack_id) -> int:
    """attackIdから基礎ダメージを引く（all_attack()を初回のみ読み込みキャッシュ）。"""
    attack = _attack_data(attack_id)
    return attack.damage if attack else 0


def _incoming_threat(obs, my_index) -> float:
    """相手の現在のエネルギーで使用可能なワザの最大実効ダメージを返す。"""
    if obs.current is None:
        return 0.0
    opponent = obs.current.players[1 - my_index]
    own = obs.current.players[my_index]
    opp_active = opponent.active or []
    my_active = own.active or []
    if not opp_active or opp_active[0] is None or not my_active or my_active[0] is None:
        return 0.0

    attacker = opp_active[0]
    defender = my_active[0]
    attacker_data = _card_data(getattr(attacker, "id", None))
    defender_data = _card_data(getattr(defender, "id", None))
    attacks = getattr(attacker_data, "attacks", None) if attacker_data else None
    if not attacks or defender_data is None:
        return 0.0

    attacker_type = getattr(attacker_data, "energyType", None)
    weakness = getattr(defender_data, "weakness", None)
    resistance = getattr(defender_data, "resistance", None)
    current_energies = getattr(attacker, "energies", None) or []
    max_damage = 0.0
    for attack_id in attacks:
        attack = _attack_data(attack_id)
        if attack is None:
            continue
        required = getattr(attack, "energies", None) or []
        if _remaining_cost(current_energies, None, required) != 0:
            continue
        damage = float(getattr(attack, "damage", 0) or 0)
        if attacker_type is not None and weakness is not None:
            if int(attacker_type) == int(weakness):
                damage *= 2
        if attacker_type is not None and resistance is not None:
            if int(attacker_type) == int(resistance):
                damage = max(0.0, damage - 30)
        max_damage = max(max_damage, damage)
    return max_damage


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    options = obs.select.option or []
    n = len(options)
    max_cnt = obs.select.maxCount
    min_cnt = obs.select.minCount
    ctx = int(obs.select.context)
    k = max(min_cnt, min(max_cnt, n))

    # SETUP_ACTIVE (1)
    if ctx == 1:
        prio = [DREEPY, DUSKULL, DUSCLOPS, FEZANDIPITI_EX, BUDEW, DRAGAPULT_EX]
        for t in prio:
            for i, opt in enumerate(options):
                if opt and _card_id(opt) == t:
                    return [i]
        return [0]

    # SETUP_BENCH (2)
    if ctx == 2:
        prio = [MEOWTH_EX, SHAYMIN, DREEPY, DUSKULL, FEZANDIPITI_EX, BUDEW, DRAKLOAK, DUSCLOPS, MUNKIDORI]
        order, used = [], set()
        for t in prio:
            for i, opt in enumerate(options):
                if i not in used and opt and _card_id(opt) == t:
                    order.append(i); used.add(i)
        for i in range(n):
            if i not in used:
                order.append(i)
        return order[:k]

    # SWITCH / TO_ACTIVE (3, 4)
    if ctx in (3, 4):
        prio = [DREEPY, DUSKULL, DUSCLOPS, BUDEW, FEZANDIPITI_EX, DUSKNOIR, DRAGAPULT_EX]
        for t in prio:
            for i, opt in enumerate(options):
                if opt and _card_id(opt) == t:
                    return [i]
        return [0]

    # TO_BENCH (5)
    if ctx == 5:
        prio = [MEOWTH_EX, SHAYMIN, DREEPY, DUSKULL, BUDEW, FEZANDIPITI_EX, MUNKIDORI]
        sel, used = [], set()
        for t in prio:
            if len(sel) >= k:
                break
            for i, opt in enumerate(options):
                if i not in used and opt and _card_id(opt) == t:
                    sel.append(i); used.add(i)
        for i in range(n):
            if len(sel) >= k:
                break
            if i not in used:
                sel.append(i)
        return sel[:k]

    # TO_HAND (7)
    if ctx == 7:
        prio = [DRAGAPULT_EX, DUSKNOIR, DRAKLOAK, DUSCLOPS, DREEPY, DUSKULL,
                FEZANDIPITI_EX, MEOWTH_EX, MUNKIDORI, SHAYMIN, BUDEW,
                RARE_CANDY, ULTRA_BALL, POKE_PAD, BUDDY_BUDDY_POFFIN,
                NIGHT_STRETCHER, TEAM_ROCKETS_PETREL, LILLIES_DETERMINATION,
                CRISPIN, BOSSS_ORDERS, UNFAIR_STAMP, JUDGE,
                TEAM_ROCKETS_WATCHTOWER, RISKY_RUINS,
                PSYCHIC_ENERGY, FIRE_ENERGY, DARKNESS_ENERGY]
        sel, used = [], set()
        for t in prio:
            if len(sel) >= k:
                break
            for i, opt in enumerate(options):
                if i not in used and opt and _card_id(opt) == t:
                    sel.append(i); used.add(i)
        for i in range(n):
            if len(sel) >= k:
                break
            if i not in used:
                sel.append(i)
        return sel[:k]

    # DISCARD (8)
    if ctx == 8:
        # DARKNESS_ENERGYは数が少なくMUNKIDORI専用のため温存する
        dprio = [FIRE_ENERGY, PSYCHIC_ENERGY, JUDGE, CRISPIN, TEAM_ROCKETS_WATCHTOWER, RISKY_RUINS]
        sel, used = [], set()
        for t in dprio:
            if len(sel) >= k:
                break
            for i, opt in enumerate(options):
                if i not in used and opt and _card_id(opt) == t:
                    sel.append(i); used.add(i); break
        for i in range(n):
            if len(sel) >= k:
                break
            if i not in used:
                sel.append(i)
        return sel[:k]

    # DAMAGE_COUNTER_ANY (14)
    if ctx == 14:
        best_i, best_s = 0, -999.0
        opp_hp_list = _opp_hps(obs)
        own_pz = len(obs.current.players[obs.current.yourIndex].prize or []) if obs.current else 6
        remain = getattr(obs.select, "remainDamageCounter", 6) or 6
        bench_candidates = []
        seen_targets = set()
        for i, opt in enumerate(options):
            if opt is None:
                continue
            area = int(getattr(opt, "area", 5))
            if area != 5:
                continue
            pi = int(getattr(opt, "playerIndex", 1 - (obs.current.yourIndex if obs.current else 0)))
            ai = int(getattr(opt, "index", 0))
            key = (pi, ai)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            p = obs.current.players[pi] if obs.current and pi < len(obs.current.players) else None
            bench = (p.bench or []) if p else []
            target = bench[ai] if ai < len(bench) else None
            hp = _hp(target)
            tdata = _card_data(getattr(target, "id", None)) if target else None
            prize_value = (
                3.0 if getattr(tdata, "megaEx", False)
                else 2.0 if getattr(tdata, "ex", False)
                else 1.0
            )
            bench_candidates.append({"option_index": i, "hp": hp, "prize_value": prize_value})

        planned_count = {}
        if bench_candidates:
            alloc = _plan_damage_distribution(
                [{"hp": c["hp"], "prize_value": c["prize_value"]} for c in bench_candidates],
                remain,
                own_pz,
            )
            for c, count in zip(bench_candidates, alloc):
                if count > 0:
                    planned_count[c["option_index"]] = count
        for i, opt in enumerate(options):
            if opt is None:
                continue
            area = int(getattr(opt, "area", 5))
            score = 10.0 if area == 5 else 0.0
            # try to find the HP of this option's pokemon
            pi = int(getattr(opt, "playerIndex", 1 - (obs.current.yourIndex if obs.current else 0)))
            ai = int(getattr(opt, "index", 0))
            if obs.current and pi < len(obs.current.players):
                p = obs.current.players[pi]
                active = p.active or []
                bench = p.bench or []
                if area == 4 and ai < len(active):
                    target = active[ai]
                    hp = _hp(target)
                elif area == 5 and ai < len(bench):
                    target = bench[ai]
                    hp = _hp(target)
                else:
                    target = None
                    hp = 100
                target_data = _card_data(getattr(target, "id", None)) if target else None
                prize_value = (
                    3.0 if getattr(target_data, "megaEx", False)
                    else 2.0 if getattr(target_data, "ex", False)
                    else 1.0
                )
                if hp <= remain * 10:
                    score += (1000.0 - hp) * prize_value
                if prize_value >= 2.0 and own_pz <= 2:
                    score -= 400.0
                score += (300 - hp) * 0.5
            if i in planned_count:
                score += 5000.0
            score += random.uniform(0, 1)
            if score > best_s:
                best_s = score; best_i = i
        return [best_i]

    # DISCARD_ENERGY (26)
    if ctx == 26:
        for i, opt in enumerate(options):
            if opt and _card_id(opt) == FIRE_ENERGY:
                return [i]
        return [0]

    # EFFECT_TARGET (25)
    if ctx == 25:
        best_i, best_score = 0, float("-inf")
        if obs.current:
            oi = 1 - obs.current.yourIndex
            opp = obs.current.players[oi]
            active = opp.active or []
            bench = opp.bench or []
            for i, opt in enumerate(options):
                if opt is None:
                    continue
                area = int(getattr(opt, "area", 4))
                ai = int(getattr(opt, "index", 0))
                if area == 4 and ai < len(active):
                    pokemon = active[ai]
                elif area == 5 and ai < len(bench):
                    pokemon = bench[ai]
                else:
                    pokemon = None
                hp = _hp(pokemon)
                data = _card_data(getattr(pokemon, "id", None)) if pokemon else None
                prize_value = 3.0 if getattr(data, "megaEx", False) else 2.0 if getattr(data, "ex", False) else 1.0
                score = prize_value * 100.0 - hp
                if score > best_score:
                    best_score = score; best_i = i
        return [best_i]

    # ATTACK (35) - ダメージ量を見て最も強い(or KOできる)ワザを選ぶ
    if ctx == 35:
        opp_active_hp = _opp_active_hp(obs)
        blocked = _opp_active_ex_immune(obs) and _my_active_is_ex(
            obs, obs.current.yourIndex if obs.current else 0
        )
        best_i, best_s = 0, -1.0
        for i, opt in enumerate(options):
            if opt is None:
                continue
            dmg = _attack_damage(getattr(opt, "attackId", None))
            score = float(dmg)
            if blocked:
                score = -500.0
            elif opp_active_hp is not None and 0 < opp_active_hp <= dmg:
                score += 500.0
            score += random.uniform(0, 1)
            if score > best_s:
                best_s = score; best_i = i
        return [best_i]

    # MAIN_PHASE (0)
    if ctx == 0:
        cur = obs.current
        opp_hps = _opp_hps(obs)
        opp_pz = len(cur.players[1 - cur.yourIndex].prize or []) if cur else 6
        own_pz = len(cur.players[cur.yourIndex].prize or []) if cur else 6
        bench_sz = len(cur.players[cur.yourIndex].bench or []) if cur else 0
        en_att = getattr(cur, "energyAttached", False) if cur else False
        sup_pl = getattr(cur, "supporterPlayed", False) if cur else False
        retreated = getattr(cur, "retreated", False) if cur else False
        own = cur.players[cur.yourIndex] if cur else None
        my_field = [
            pokemon
            for pokemon in ((own.active or []) + (own.bench or []) if own else [])
            if pokemon is not None
        ]
        have_ready_attacker = False
        ex_attacker_ready = False
        for pokemon in my_field:
            pokemon_data = _card_data(getattr(pokemon, "id", None))
            attacks = getattr(pokemon_data, "attacks", None) if pokemon_data else None
            current_energies = getattr(pokemon, "energies", None) or []
            for attack_id in attacks or []:
                attack = _attack_data(attack_id)
                if attack is None:
                    continue
                required = getattr(attack, "energies", None) or []
                if _remaining_cost(current_energies, None, required) == 0:
                    have_ready_attacker = True
                    if getattr(pokemon_data, "ex", False) or getattr(
                        pokemon_data, "megaEx", False
                    ):
                        ex_attacker_ready = True
                    break
            if have_ready_attacker and ex_attacker_ready:
                break
        crustle_on_bench = _opp_bench_has_ex_immune(obs)
        dragapult_count = sum(
            1 for pokemon in my_field if getattr(pokemon, "id", None) == DRAGAPULT_EX
        )

        scored = []
        for i, opt in enumerate(options):
            if opt is None:
                scored.append((random.uniform(0, 1), i))
                continue
            ot = _opt_type(opt)
            cid = _card_id(opt)
            score = 0.0

            if ot == OPT_ATTACK:
                dmg = _attack_damage(getattr(opt, "attackId", None))
                score = 100.0 + dmg * 0.3
                opp_active_hp = _opp_active_hp(obs)
                if opp_active_hp is not None and 0 < opp_active_hp <= dmg:
                    score += 200.0
                if opp_pz <= 2:
                    score += 60.0
                if _opp_active_ex_immune(obs) and _my_active_is_ex(
                    obs, cur.yourIndex if cur else 0
                ):
                    score = -50.0

            elif ot == OPT_ABILITY:
                score = 80.0
                blocked_wall = _opp_active_ex_immune(obs)
                if cid == DUSKNOIR:
                    if any(0 < hp <= DUSKNOIR_DMG for hp in opp_hps):
                        score += 200.0
                    elif blocked_wall:
                        score += 50.0
                    else:
                        score -= 30.0
                elif cid == DUSCLOPS:
                    if any(0 < hp <= DUSCLOPS_DMG for hp in opp_hps):
                        score += 150.0
                    elif blocked_wall:
                        score += 30.0
                    else:
                        score -= 20.0

            elif ot == OPT_EVOLVE:
                prio = {DRAGAPULT_EX: 75, DUSKNOIR: 70, DRAKLOAK: 55, DUSCLOPS: 50}
                score = prio.get(cid, 60.0)
                if cid == DRAGAPULT_EX and (
                    dragapult_count >= 2 or (dragapult_count == 1 and opp_pz <= 2)
                ):
                    score = -30.0

            elif ot == OPT_PLAY:
                if cid == MEOWTH_EX:
                    # ベンチに出すだけでサポーターをサーチできる
                    score = 95.0
                elif cid == SHAYMIN:
                    # 常時能力でベンチを守れるので早めに展開する
                    score = 90.0 if bench_sz == 0 else 70.0
                elif cid in (DREEPY, DUSKULL, FEZANDIPITI_EX, BUDEW):
                    score = 40.0
                    if bench_sz == 0:
                        score += 120.0
                    elif bench_sz <= 2:
                        score += 30.0
                elif cid == MUNKIDORI:
                    score = 35.0
                elif cid == UNFAIR_STAMP:
                    score = 90.0
                elif cid == BOSSS_ORDERS:
                    ko = [hp for hp in opp_hps if 0 < hp <= 130]
                    score = 85.0 + len(ko) * 20
                elif cid == LILLIES_DETERMINATION and sup_pl:
                    score = -50.0
                elif cid == LILLIES_DETERMINATION:
                    # 自分のプライズが6枚(まだ取られていない)なら8枚ドローになる
                    score = 78.0 if own_pz == 6 else 45.0
                elif cid in (TEAM_ROCKETS_PETREL, CRISPIN, JUDGE) and sup_pl:
                    score = -50.0
                elif cid == TEAM_ROCKETS_PETREL:
                    score = 68.0
                elif cid == CRISPIN:
                    score = 65.0 if not en_att else 50.0
                elif cid == JUDGE:
                    score = 60.0
                elif cid == ULTRA_BALL:
                    score = 60.0
                elif cid == POKE_PAD:
                    score = 58.0
                elif cid == RARE_CANDY:
                    score = 55.0
                elif cid == NIGHT_STRETCHER:
                    score = 52.0
                elif cid == BUDDY_BUDDY_POFFIN:
                    score = 50.0 if bench_sz <= 3 else 20.0
                elif cid in (TEAM_ROCKETS_WATCHTOWER, RISKY_RUINS):
                    score = 35.0
                else:
                    score = 20.0

            elif ot == OPT_ATTACH:
                my_index = cur.yourIndex if cur else 0
                src = _get_card_or_pokemon(
                    obs,
                    getattr(opt, "area", None),
                    int(getattr(opt, "index", 0) or 0),
                    my_index,
                )
                src_data = _card_data(getattr(src, "id", None)) if src else None
                src_type = getattr(src_data, "energyType", None) if src_data else None
                target = _get_card_or_pokemon(
                    obs,
                    getattr(opt, "inPlayArea", None),
                    int(getattr(opt, "inPlayIndex", 0) or 0),
                    my_index,
                )
                target_data = _card_data(getattr(target, "id", None)) if target else None
                attacks = getattr(target_data, "attacks", None) if target_data else None
                target_is_ex = bool(
                    target_data
                    and (
                        getattr(target_data, "ex", False)
                        or getattr(target_data, "megaEx", False)
                    )
                )

                if target is None or not attacks:
                    score = 20.0
                else:
                    before = 99
                    after = 99
                    current_energies = getattr(target, "energies", None) or []
                    for attack_id in attacks:
                        attack = _attack_data(attack_id)
                        if attack is None:
                            continue
                        required = getattr(attack, "energies", None) or []
                        before = min(before, _remaining_cost(current_energies, None, required))
                        after = min(after, _remaining_cost(current_energies, src_type, required))
                    score = 40.0
                    if after == 0 and before > 0:
                        score += 200.0
                    else:
                        progress = max(0, before - after) * 30.0
                        if have_ready_attacker:
                            progress *= 0.3
                        score += progress
                    if before == 0:
                        score -= 150.0
                    if before == 99:
                        score -= 30.0
                    if getattr(opt, "inPlayArea", None) == AreaType.ACTIVE:
                        score += 15.0
                if (
                    crustle_on_bench
                    and ex_attacker_ready
                    and target is not None
                    and not target_is_ex
                ):
                    score += 120.0
                if en_att:
                    score -= 50.0

            elif ot == OPT_RETREAT:
                if retreated:
                    score = -100.0
                else:
                    cost = int(getattr(opt, "count", 2) or 2)
                    my_index = cur.yourIndex if cur else 0
                    threat = _incoming_threat(obs, my_index)
                    active = cur.players[my_index].active or [] if cur else []
                    active_hp = _hp(active[0]) if active and active[0] is not None else 9999
                    if threat > 0 and active_hp <= threat:
                        score = 150.0 - cost * 5
                    else:
                        score = 30.0 - cost * 10

            elif ot == OPT_END:
                score = -30.0

            score += random.uniform(0, 1)
            scored.append((score, i))

        scored.sort(reverse=True)
        return [scored[0][1]]

    # default random
    return random.sample(list(range(n)), k)
