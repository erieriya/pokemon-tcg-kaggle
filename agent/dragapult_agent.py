import os
import random

from cg.api import Observation, to_observation_class

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


_ATTACK_DB: dict[int, object] | None = None


def _attack_damage(attack_id) -> int:
    """attackIdから基礎ダメージを引く（all_attack()を初回のみ読み込みキャッシュ）。"""
    global _ATTACK_DB
    if _ATTACK_DB is None:
        from cg.api import all_attack
        _ATTACK_DB = {a.attackId: a for a in all_attack()}
    attack = _ATTACK_DB.get(attack_id)
    return attack.damage if attack else 0


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
        remain = getattr(obs.select, "remainDamageCounter", 6) or 6
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
                    hp = _hp(active[ai])
                elif area == 5 and ai < len(bench):
                    hp = _hp(bench[ai])
                else:
                    hp = 100
                if hp <= remain * 10:
                    score += 1000.0 - hp
                score += (300 - hp) * 0.5
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
        best_i, best_hp = 0, 9999
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
                    hp = _hp(active[ai])
                elif area == 5 and ai < len(bench):
                    hp = _hp(bench[ai])
                else:
                    hp = 9999
                if hp < best_hp:
                    best_hp = hp; best_i = i
        return [best_i]

    # ATTACK (35) - ダメージ量を見て最も強い(or KOできる)ワザを選ぶ
    if ctx == 35:
        opp_active_hp = _opp_active_hp(obs)
        best_i, best_s = 0, -1.0
        for i, opt in enumerate(options):
            if opt is None:
                continue
            dmg = _attack_damage(getattr(opt, "attackId", None))
            score = float(dmg)
            if opp_active_hp is not None and 0 < opp_active_hp <= dmg:
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

            elif ot == OPT_ABILITY:
                score = 80.0
                if cid == DUSKNOIR:
                    if any(0 < hp <= DUSKNOIR_DMG for hp in opp_hps):
                        score += 200.0
                    else:
                        score -= 30.0
                elif cid == DUSCLOPS:
                    if any(0 < hp <= DUSCLOPS_DMG for hp in opp_hps):
                        score += 150.0
                    else:
                        score -= 20.0

            elif ot == OPT_EVOLVE:
                prio = {DRAGAPULT_EX: 75, DUSKNOIR: 70, DRAKLOAK: 55, DUSCLOPS: 50}
                score = prio.get(cid, 60.0)

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
                score = 45.0 if not en_att else -5.0

            elif ot == OPT_RETREAT:
                if retreated:
                    score = -100.0
                else:
                    cost = int(getattr(opt, "count", 2) or 2)
                    score = 30.0 - cost * 10

            elif ot == OPT_END:
                score = -30.0

            score += random.uniform(0, 1)
            scored.append((score, i))

        scored.sort(reverse=True)
        return [scored[0][1]]

    # default random
    return random.sample(list(range(n)), k)
