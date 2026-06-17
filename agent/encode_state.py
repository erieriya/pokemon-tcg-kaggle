"""
cabt Engine API に基づいた状態エンコーダ

API型定義:
  EnergyType: COLORLESS=0, GRASS=1, FIRE=2, WATER=3, LIGHTNING=4,
              PSYCHIC=5, FIGHTING=6, DARKNESS=7, METAL=8, DRAGON=9,
              RAINBOW=10, TEAM_ROCKET=11

  Pokemon: id, serial, hp, maxHp, appearThisTurn,
           energies(list[EnergyType]), energyCards, tools, preEvolution

  PlayerState: active, bench, hand, discard, prize,
               deckCount, handCount, benchMax,
               poisoned, burned, asleep, paralyzed, confused

  State: turn, yourIndex, firstPlayer, players, stadium,
         supporterPlayed, stadiumPlayed, energyAttached, retreated, result

  Option: type(OptionType), attackId, cardId, serial,
          inPlayArea, inPlayIndex, count, energyIndex
"""

import torch

N_ENERGY_TYPES = 12  # COLORLESS〜TEAM_ROCKET
N_STATUS = 5         # poison, burn, sleep, paralyze, confuse
MAX_CARD_ID = 5000
CARD_EMBED_DIM = 128
POKEMON_FEAT_DIM = 32  # ポケモン1体の特徴量次元
PLAYER_FEAT_DIM = 64   # プレイヤー状態の特徴量次元
GLOBAL_FEAT_DIM = 16   # グローバル特徴量次元
ACTION_FEAT_DIM = 32   # アクション特徴量次元

# OptionType 定数
OPT_NUMBER = 0; OPT_YES = 1; OPT_NO = 2; OPT_CARD = 3
OPT_TOOL_CARD = 4; OPT_ENERGY_CARD = 5; OPT_ENERGY = 6
OPT_PLAY = 7; OPT_ATTACH = 8; OPT_EVOLVE = 9; OPT_ABILITY = 10
OPT_DISCARD = 11; OPT_RETREAT = 12; OPT_ATTACK = 13
OPT_END = 14; OPT_SKILL = 15; OPT_SPECIAL_CONDITION = 16
N_OPT_TYPES = 17


def _get(obj, *keys, default=None):
    """dict/dataclass の安全なフィールド取得"""
    for key in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj if obj is not None else default


def encode_pokemon(poke) -> list[float]:
    """Pokemon オブジェクトを固定長の特徴量ベクトルに変換"""
    if poke is None:
        return [0.0] * POKEMON_FEAT_DIM

    hp = _get(poke, "hp", default=0)
    max_hp = _get(poke, "maxHp", default=100)
    hp_ratio = hp / max(max_hp, 1)

    # エネルギー分布 (12タイプ)
    energies = _get(poke, "energies", default=[])
    energy_vec = [0.0] * N_ENERGY_TYPES
    for e in energies:
        idx = int(e) if isinstance(e, (int, float)) else 0
        if 0 <= idx < N_ENERGY_TYPES:
            energy_vec[idx] += 1.0 / 8.0  # 正規化（最大8個想定）

    # ツール・進化前カードの有無
    has_tool = float(len(_get(poke, "tools", default=[])) > 0)
    has_pre_evo = float(len(_get(poke, "preEvolution", default=[])) > 0)
    appeared_this_turn = float(_get(poke, "appearThisTurn", default=False))

    feats = [
        hp_ratio,
        float(hp) / 300.0,       # 絶対HP
        float(max_hp) / 300.0,   # 最大HP
        float(len(energies)) / 8.0,
        has_tool,
        has_pre_evo,
        appeared_this_turn,
    ] + energy_vec + [0.0] * (POKEMON_FEAT_DIM - 7 - N_ENERGY_TYPES)

    return feats[:POKEMON_FEAT_DIM]


def encode_player(player, is_self: bool = True) -> list[float]:
    """PlayerState を特徴量ベクトルに変換"""
    if player is None:
        return [0.0] * PLAYER_FEAT_DIM

    active_list = _get(player, "active", default=[])
    bench = _get(player, "bench", default=[])
    prize = _get(player, "prize", default=[])
    deck_count = _get(player, "deckCount", default=0)
    hand_count = _get(player, "handCount", default=0)
    bench_max = _get(player, "benchMax", default=5)

    # 状態異常フラグ
    status_feats = [
        float(_get(player, "poisoned", default=False)),
        float(_get(player, "burned", default=False)),
        float(_get(player, "asleep", default=False)),
        float(_get(player, "paralyzed", default=False)),
        float(_get(player, "confused", default=False)),
    ]

    # サイドカード
    prize_remaining = sum(1 for p in prize if p is not None)
    prize_face_down = sum(1 for p in prize if p is None)

    # ベンチのエネルギー合計
    bench_energy_total = sum(
        len(_get(p, "energies", default=[]))
        for p in bench
        if p is not None
    )

    feats = [
        float(len(active_list)) / 1.0,
        float(len(bench)) / bench_max,
        float(prize_remaining) / 6.0,
        float(prize_face_down) / 6.0,
        float(deck_count) / 60.0,
        float(hand_count) / 20.0,
        float(bench_energy_total) / 20.0,
    ] + status_feats + [0.0] * (PLAYER_FEAT_DIM - 7 - N_STATUS)

    return feats[:PLAYER_FEAT_DIM]


def encode_action(opt) -> list[float]:
    """Option オブジェクトをアクション特徴量ベクトルに変換"""
    opt_type = int(_get(opt, "type", default=OPT_END))
    feats = [0.0] * N_OPT_TYPES
    if 0 <= opt_type < N_OPT_TYPES:
        feats[opt_type] = 1.0

    damage = float(_get(opt, "damage", default=0)) / 300.0
    count = float(_get(opt, "count", default=0)) / 8.0
    energy_cost = float(_get(opt, "energyIndex", default=0)) / 8.0

    extra = [damage, count, energy_cost] + [0.0] * (ACTION_FEAT_DIM - N_OPT_TYPES - 3)
    return (feats + extra)[:ACTION_FEAT_DIM]


def encode_state_tensor(obs_dict: dict, card_id_map: dict | None = None) -> dict[str, torch.Tensor]:
    """
    obs_dict を PyTorch テンソルの辞書に変換する。

    Args:
        obs_dict: cabt Engine からの観測辞書
        card_id_map: カード名→int IDのマッピング（Noneなら生のcard.idを使用）

    Returns:
        {
          "my_active_id": (1,),  # カードID
          "opp_active_id": (1,),
          "hand_ids": (20,),     # パディングあり
          "bench_ids": (5,),
          "my_player_feats": (PLAYER_FEAT_DIM,),
          "opp_player_feats": (PLAYER_FEAT_DIM,),
          "global_feats": (GLOBAL_FEAT_DIM,),
          "action_feats": (n_options, ACTION_FEAT_DIM),
        }
    """
    current = _get(obs_dict, "current")
    players = _get(current, "players", default=[])
    my = players[0] if players else None
    opp = players[1] if len(players) > 1 else None

    my_active_list = _get(my, "active", default=[])
    my_active = my_active_list[0] if my_active_list else None
    opp_active_list = _get(opp, "active", default=[])
    opp_active = opp_active_list[0] if opp_active_list else None

    def get_card_id(card) -> int:
        if card is None:
            return 0
        raw_id = int(_get(card, "id", default=0))
        if card_id_map:
            return card_id_map.get(raw_id, raw_id)
        return raw_id

    # アクティブポケモンのカードID
    my_active_id = get_card_id(my_active)
    opp_active_id = get_card_id(opp_active)

    # 手札のカードID（最大20枚、パディング0）
    hand = _get(my, "hand", default=[]) or []
    hand_ids = [get_card_id(c) for c in hand[:20]]
    hand_ids += [0] * (20 - len(hand_ids))

    # ベンチのカードID（最大5枚）
    bench = _get(my, "bench", default=[]) or []
    bench_ids = [get_card_id(p) for p in bench[:5]]
    bench_ids += [0] * (5 - len(bench_ids))

    # プレイヤー特徴量
    my_feats = encode_player(my, is_self=True)
    opp_feats = encode_player(opp, is_self=False)

    # グローバル特徴量
    turn = float(_get(current, "turn", default=0)) / 40.0
    my_prize = len([p for p in _get(my, "prize", default=[]) if p is None]) / 6.0
    opp_prize = len([p for p in _get(opp, "prize", default=[]) if p is None]) / 6.0
    supporter_played = float(_get(current, "supporterPlayed", default=False))
    energy_attached = float(_get(current, "energyAttached", default=False))
    retreated = float(_get(current, "retreated", default=False))
    stadium_present = float(len(_get(current, "stadium", default=[])) > 0)

    global_feats = [
        turn, my_prize, opp_prize,
        supporter_played, energy_attached, retreated, stadium_present,
    ] + [0.0] * (GLOBAL_FEAT_DIM - 7)

    # アクション特徴量
    select = _get(obs_dict, "select", default={}) or {}
    options = _get(select, "option", default=[]) or []
    action_feats = [encode_action(opt) for opt in options] if options else [[0.0] * ACTION_FEAT_DIM]

    return {
        "my_active_id": torch.tensor([my_active_id], dtype=torch.long),
        "opp_active_id": torch.tensor([opp_active_id], dtype=torch.long),
        "hand_ids": torch.tensor(hand_ids, dtype=torch.long),
        "bench_ids": torch.tensor(bench_ids, dtype=torch.long),
        "my_player_feats": torch.tensor(my_feats, dtype=torch.float32),
        "opp_player_feats": torch.tensor(opp_feats, dtype=torch.float32),
        "global_feats": torch.tensor(global_feats[:GLOBAL_FEAT_DIM], dtype=torch.float32),
        "action_feats": torch.tensor(action_feats, dtype=torch.float32),
    }
