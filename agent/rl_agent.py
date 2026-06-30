"""
PTCG AI Battle Challenge - PPO RL Agent (スケルトン実装)

アーキテクチャ:
  - カード埋め込み（Embedding + MLP）
  - ゲーム状態エンコーダ（構造化特徴量）
  - Policy head（legal actionsに対してAttention）
  - Value head（状態価値推定）

学習:
  - PPO (Proximal Policy Optimization)
  - 自己対戦（Self-play）
  - エントロピー正則化（β=0.05）

NOTE: cg/api.pyを読んだ後に状態エンコーダの実装を確定させること。
      カードIDとフィールド名はEN_Card_Data.csvに合わせること。
"""

import json
import math
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kaggle実行環境では、main.py初回import後にcwdやsys.pathの一時エントリが
# 失われることがあり、_load_card_db/_load_attack_db内の遅延importする
# `from cg.api import ...` がModuleNotFoundErrorになる。
# __file__基準の絶対パスをsys.pathへ追加して回避する(重複チェックなしで毎回追加。
# importはモジュールロード時に一度しか実行されないので増殖しないし、既存エントリが
# 後から削除されても自分の追加分は残る)。
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AGENT_DIR)

# カードIDの最大値（暫定: EN_Card_Data.csvを確認後に更新）
MAX_CARD_ID = 3000
EMBED_DIM = 128
STATE_DIM = 256
HIDDEN_DIM = 256
N_ENERGY_TYPES = 12  # cg.api.EnergyType: COLORLESS(0)〜TEAM_ROCKET(11)
N_OPTION_TYPES = 17  # cg.api.OptionType: NUMBER(0)〜SPECIAL_CONDITION(16)
EX_DAMAGE_IMMUNE_IDS = {345}  # Crustle「Mysterious Rock Inn」。dragapult_agent_v2.pyの同名定数と同じ
DAMAGE_COUNTER_IMMUNE_ENERGY_IDS = {11, 20}  # Mist Energy, Rock Fighting Energy。dragapult_agent_v2.pyの同名定数と同じ
POKE_SCALAR_DIM = 2 + N_ENERGY_TYPES + 5 + 4 + 1 + 4 + 1 + 3 + N_ENERGY_TYPES + 1 + 1 + 1 + 1 + 1
# 内訳: hp_ratio+dmg(2) + エネルギー(12) + 状態異常(5) + 静的特徴(4) + 被ダメージ(1) + 進化脅威(4)
#       + ツール(1) + 特性関連(3) + card_energy_type 1-hot(12) + weakness(1) + resistance(1)
#       + can_attack_now(1) + max_outgoing_damage(1) + appear_this_turn(1)
AREA_HAND = 2  # cg.api.AreaType.HAND
AREA_ACTIVE = 4  # cg.api.AreaType.ACTIVE
AREA_BENCH = 5  # cg.api.AreaType.BENCH
OPP_DISCARD_CAP = 60


class CardEmbedding(nn.Module):
    """カードIDを埋め込みベクトルに変換"""

    def __init__(self, max_card_id: int = MAX_CARD_ID, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(max_card_id + 1, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(card_ids)
        return F.relu(self.proj(x))


class HandEncoder(nn.Module):
    """手札全体をAttention Poolingでエンコード"""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, hand_embeds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = hand_embeds.size(0)
        q = self.pool_query.expand(B, -1, -1)
        out, _ = self.attn(q, hand_embeds, hand_embeds, key_padding_mask=mask)
        return out.squeeze(1)


class PokemonEncoder(nn.Module):
    """場のポケモン状態をエンコード（HP比率・エネルギー・ダメカン等）"""

    def __init__(self, embed_dim: int = EMBED_DIM, out_dim: int = HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + POKE_SCALAR_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim),
        )

    def forward(self, card_embed: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        # card_embed: (batch, 1, embed_dim) → squeeze to (batch, embed_dim)
        if card_embed.dim() == 3:
            card_embed = card_embed.squeeze(1)
        x = torch.cat([card_embed, scalar_features], dim=-1)
        return self.mlp(x)


class StateEncoder(nn.Module):
    """ゲーム全体の状態をエンコード"""

    def __init__(self):
        super().__init__()
        self.card_emb = CardEmbedding()
        self.hand_enc = HandEncoder()
        self.poke_enc = PokemonEncoder()
        # hand/stadium/opp_discard(EMBED_DIM) + active/bench(HIDDEN_DIM) + scalars(24)
        concat_dim = EMBED_DIM * 3 + HIDDEN_DIM * 4 + 24
        self.global_proj = nn.Linear(concat_dim, STATE_DIM)
        self.output_norm = nn.LayerNorm(STATE_DIM)

    def _pool_bench(
        self,
        ids: torch.Tensor,
        scalar: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """ベンチ各個体をPokemonEncoderに通し、マスク付き平均で集約する。"""
        embeds = self.card_emb(ids)
        B, K, E = embeds.shape
        poke_vec = self.poke_enc(
            embeds.reshape(B * K, 1, E),
            scalar.reshape(B * K, -1),
        ).reshape(B, K, -1)
        expanded_mask = mask.unsqueeze(-1)
        return (poke_vec * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1.0)

    def _pool_embeddings(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """カード埋め込みをマスク付き平均で集約する。"""
        embeds = self.card_emb(ids)
        expanded_mask = mask.unsqueeze(-1)
        return (embeds * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1.0)

    def forward(self, state: dict) -> torch.Tensor:
        """
        Args:
            state: encode_state()の出力（テンソル形式）
        Returns:
            state_vec: (batch, STATE_DIM)
        """
        # 手札
        hand_ids = state["hand_ids"]
        hand_embs = self.card_emb(hand_ids)
        hand_vec = self.hand_enc(hand_embs)

        # 自分のアクティブポケモン
        my_active_id = state["my_active_id"]
        my_active_emb = self.card_emb(my_active_id)
        my_active_vec = self.poke_enc(my_active_emb, state["my_active_scalar"])

        # 相手のアクティブポケモン
        opp_active_id = state["opp_active_id"]
        opp_active_emb = self.card_emb(opp_active_id)
        opp_active_vec = self.poke_enc(opp_active_emb, state["opp_active_scalar"])

        # ベンチ
        bench_vec = self._pool_bench(
            state["bench_ids"],
            state["bench_scalar"],
            state["bench_mask"],
        )
        opp_bench_vec = self._pool_bench(
            state["opp_bench_ids"],
            state["opp_bench_scalar"],
            state["opp_bench_mask"],
        )

        # スタジアムと相手の捨て札
        stadium_vec = self.card_emb(state["stadium_id"]).squeeze(1)
        opp_discard_vec = self._pool_embeddings(
            state["opp_discard_ids"],
            state["opp_discard_mask"],
        )

        # グローバル情報（サイド枚数、ターン数等）
        global_scalars = state["global_scalars"]

        combined = torch.cat([
            hand_vec,
            my_active_vec,
            opp_active_vec,
            bench_vec,
            opp_bench_vec,
            stadium_vec,
            opp_discard_vec,
            global_scalars,
        ], dim=-1)
        return self.output_norm(F.relu(self.global_proj(combined)))


class PTCGNet(nn.Module):
    """PolicyとValueを出力するネットワーク"""

    def __init__(self, max_actions: int = 100):
        super().__init__()
        self.state_enc = StateEncoder()
        self.card_emb = CardEmbedding()

        # action_features (EMBED_DIM) を STATE_DIM に投影
        self.action_proj = nn.Linear(EMBED_DIM, STATE_DIM)
        self.action_norm = nn.LayerNorm(STATE_DIM)
        self.policy_attn = nn.MultiheadAttention(
            embed_dim=STATE_DIM, num_heads=4, batch_first=True
        )
        self.policy_norm = nn.LayerNorm(STATE_DIM)
        self.policy_head = nn.Linear(STATE_DIM, 1)
        self.value_head = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        nn.init.uniform_(self.value_head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        state: dict,
        action_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: encode_state()の出力
            action_features: (batch, n_actions, EMBED_DIM) 各行動の特徴量
            action_mask: (batch, n_actions) Trueが有効な行動
        Returns:
            logits: (batch, n_actions)
            value: (batch, 1)
        """
        state_vec = self.state_enc(state)

        # action_features を STATE_DIM に投影
        action_proj = self.action_norm(
            self.action_proj(action_features)
        )  # (B, n_actions, STATE_DIM)

        q = state_vec.unsqueeze(1)  # (B, 1, STATE_DIM)
        key_padding_mask = None
        if action_mask is not None:
            key_padding_mask = ~action_mask.bool()
        attn_out, _ = self.policy_attn(
            q, action_proj, action_proj, key_padding_mask=key_padding_mask
        )
        logits = self.policy_head(self.policy_norm(attn_out + action_proj)).squeeze(
            -1
        )  # (B, n_actions)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

        value = torch.tanh(self.value_head(state_vec))
        return logits, value


_CARD_DB: dict[int, object] | None = None
_ATTACK_DB: dict[int, object] | None = None
_EVOLUTION_INDEX: dict | None = None


def _load_card_db() -> dict:
    global _CARD_DB
    if _CARD_DB is None:
        from cg.api import all_card_data
        _CARD_DB = {c.cardId: c for c in all_card_data()}
    return _CARD_DB


def _load_attack_db() -> dict:
    global _ATTACK_DB
    if _ATTACK_DB is None:
        from cg.api import all_attack
        _ATTACK_DB = {a.attackId: a for a in all_attack()}
    return _ATTACK_DB


def _load_evolution_index() -> dict:
    """カード名 -> そのカードから進化するCardDataのリスト、の逆引き辞書。"""
    global _EVOLUTION_INDEX
    if _EVOLUTION_INDEX is None:
        index = {}
        for card in _load_card_db().values():
            if card.evolvesFrom is not None:
                index.setdefault(card.evolvesFrom, []).append(card)
        _EVOLUTION_INDEX = index
    return _EVOLUTION_INDEX


def _future_evolutions(card, evolution_index, max_depth: int = 3) -> list:
    """cardから深さmax_depthまでに到達できる将来の進化先を返す。"""
    if card is None:
        return []
    result = []
    seen_ids = {card.cardId}
    frontier = {card.name}
    for _ in range(max_depth):
        next_frontier = set()
        for name in frontier:
            for evolved in evolution_index.get(name, []):
                if evolved.cardId in seen_ids:
                    continue
                seen_ids.add(evolved.cardId)
                result.append(evolved)
                next_frontier.add(evolved.name)
        frontier = next_frontier
        if not frontier:
            break
    return result


def _evolution_threat_feats(attack_db: dict, evolution_index: dict, card) -> list[float]:
    """将来の進化先が持つ最大HP・最大打点・ex化を集約して返す。"""
    futures = _future_evolutions(card, evolution_index)
    if not futures:
        return [0.0, 0.0, 0.0, 0.0]
    max_hp = max(float(getattr(evolved, "hp", 0) or 0) for evolved in futures)
    max_damage = 0.0
    for evolved in futures:
        for attack_id in getattr(evolved, "attacks", None) or []:
            attack = attack_db.get(attack_id)
            if attack is not None:
                max_damage = max(max_damage, float(getattr(attack, "damage", 0) or 0))
    will_become_ex = any(
        getattr(evolved, "ex", False) or getattr(evolved, "megaEx", False)
        for evolved in futures
    )
    return [1.0, max_hp / 300.0, max_damage / 300.0, float(will_become_ex)]


def _energy_count_vec(energies: list | None) -> list[float]:
    """エネルギーのリストをタイプ別カウントの生ベクトル(正規化なし)に変換"""
    vec = [0.0] * N_ENERGY_TYPES
    for e in energies or []:
        idx = int(e)
        if 0 <= idx < N_ENERGY_TYPES:
            vec[idx] += 1.0
    return vec


def _resolve_card(obs_dict: dict, area: int | None, index: int | None, player_idx: int):
    """Option.area/index (または inPlayArea/inPlayIndex) からカード/ポケモンのdictを引く。
    cg Engineの生dict版（lucario_v1_agent.get_cardのObservationクラス版に相当）。
    """
    if area is None or index is None:
        return None
    try:
        player = (obs_dict.get("current") or {}).get("players", [])[player_idx]
    except (IndexError, TypeError):
        return None
    try:
        if area == AREA_HAND:
            return (player.get("hand") or [])[index]
        if area == AREA_ACTIVE:
            return (player.get("active") or [])[index]
        if area == AREA_BENCH:
            return (player.get("bench") or [])[index]
    except (IndexError, TypeError):
        return None
    return None


def _remaining_energy_cost(current: list[int], extra_type: int | None, required: list[int]) -> int:
    """current(+extra_typeを仮に追加)がrequiredをどれだけ満たせていないか(不足数)を返す。
    無色(COLORLESS=0)要求は色指定要求を満たした後の余りエネルギーで埋められる。
    """
    pool = list(current)
    if extra_type is not None:
        pool.append(extra_type)
    colorless_needed = 0
    missing = 0
    for req in required:
        if req == 0:  # COLORLESS
            colorless_needed += 1
            continue
        if req in pool:
            pool.remove(req)
        else:
            missing += 1
    missing += max(0, colorless_needed - len(pool))
    return missing


def _max_affordable_damage(
    card_db: dict,
    attack_db: dict,
    attacker: dict,
    defender_weakness,
    defender_resistance,
) -> float:
    """attacker(Pokemonのdict)が現在のエネルギーで実際に打てるワザの中で、
    defender視点の弱点/抵抗(EnergyType int|None)を加味した最大実効ダメージを返す。
    打てるワザが無ければ0.0。weakness/resistanceの判定はattacker自身のCardData.energyTypeを使う。
    """
    if not attacker:
        return 0.0
    card = card_db.get(attacker.get("id") or attacker.get("cardId"))
    attacks = getattr(card, "attacks", None) if card else None
    if not attacks:
        return 0.0
    attacker_type = getattr(card, "energyType", None)
    attacker_type = int(attacker_type) if attacker_type is not None else None
    current = [int(e) for e in (attacker.get("energies") or [])]
    max_damage = 0.0
    for attack_id in attacks:
        attack = attack_db.get(attack_id)
        if attack is None:
            continue
        required = [int(e) for e in attack.energies]
        if _remaining_energy_cost(current, None, required) != 0:
            continue
        effective_damage = float(attack.damage)
        if attacker_type is not None and defender_weakness is not None:
            if attacker_type == int(defender_weakness):
                effective_damage *= 2
        if attacker_type is not None and defender_resistance is not None:
            if attacker_type == int(defender_resistance):
                effective_damage = max(0.0, effective_damage - 30)
        max_damage = max(max_damage, effective_damage)
    return max_damage


def _attach_target_features(card_db: dict, attack_db: dict, target: dict, extra_type: int | None) -> tuple[float, float]:
    """ATTACH対象ポケモンに対し、この1枚を貼ったら(1)ワザが新たに使用可能になるか、
    (2)最も近いワザを使うのに残り何枚エネルギーが必要か、を返す。
    """
    if not target:
        return 0.0, 0.0
    card = card_db.get(target.get("id") or target.get("cardId"))
    attacks = getattr(card, "attacks", None) if card else None
    if not attacks:
        return 0.0, 0.0
    current = [int(e) for e in (target.get("energies") or [])]
    unlocks = False
    best_remaining = None
    for attack_id in attacks:
        attack = attack_db.get(attack_id)
        if attack is None:
            continue
        required = [int(e) for e in attack.energies]
        remaining = _remaining_energy_cost(current, extra_type, required)
        if remaining == 0:
            unlocks = True
        if best_remaining is None or remaining < best_remaining:
            best_remaining = remaining
    return (1.0 if unlocks else 0.0), float(best_remaining or 0)


def _card_static_feats(card_id: int) -> list[float]:
    """retreatCost / ex / megaEx / 進化stage をCardDataから取得（無ければ既定値）"""
    card = _load_card_db().get(card_id)
    if card is None:
        return [1.0, 0.0, 0.0, 0.0]
    stage = 2.0 if getattr(card, "stage2", False) else (1.0 if getattr(card, "stage1", False) else 0.0)
    return [
        float(getattr(card, "retreatCost", 1) or 0),
        float(getattr(card, "ex", False)),
        float(getattr(card, "megaEx", False)),
        stage,
    ]


def encode_state(obs_dict: dict, device: str = "cpu") -> dict:
    """obs_dict(cg.gameの生のdict)をPTCGNetの入力形式に変換するプリプロセッサ。"""
    current = obs_dict.get("current", {}) or {}
    players = current.get("players", [{}, {}]) or [{}, {}]
    your_idx = current.get("yourIndex", 0) or 0
    my = players[your_idx] if your_idx < len(players) else {}
    opp = players[1 - your_idx] if (1 - your_idx) < len(players) else {}
    card_db = _load_card_db()
    attack_db = _load_attack_db()
    evolution_index = _load_evolution_index()

    def get_card_id(card) -> int:
        if card is None:
            return 0
        if isinstance(card, dict):
            return card.get("id", card.get("cardId", 0)) or 0
        return int(card)

    def get_pokemon_scalar(
        poke: dict,
        status: list[float] | None = None,
        threat_active: dict | None = None,
    ) -> list[float]:
        if not poke:
            return [0.0] * POKE_SCALAR_DIM
        cur_hp = float(poke.get("hp", 0) or 0)
        max_hp = float(poke.get("maxHp", 0) or cur_hp or 100)
        dmg = max(0.0, max_hp - cur_hp)
        hp_ratio = cur_hp / max_hp if max_hp > 0 else 0.0

        energy_vec = [v / 4.0 for v in _energy_count_vec(poke.get("energies"))]

        status = status or [0.0] * 5
        card_feats = _card_static_feats(get_card_id(poke))
        card = card_db.get(poke.get("id") or poke.get("cardId"))
        incoming_max_damage = 0.0
        if threat_active:
            weakness = getattr(card, "weakness", None) if card else None
            resistance = getattr(card, "resistance", None) if card else None
            incoming_max_damage = _max_affordable_damage(
                card_db,
                attack_db,
                threat_active,
                weakness,
                resistance,
            )
        evolution_feats = _evolution_threat_feats(attack_db, evolution_index, card)
        has_tool = 1.0 if poke.get("tools") else 0.0
        has_ability = 1.0 if card and getattr(card, "skills", None) else 0.0
        is_ex_damage_immune = 1.0 if get_card_id(poke) in EX_DAMAGE_IMMUNE_IDS else 0.0
        has_damage_counter_immune_energy = (
            1.0
            if any(
                get_card_id(energy_card) in DAMAGE_COUNTER_IMMUNE_ENERGY_IDS
                for energy_card in (poke.get("energyCards") or [])
            )
            else 0.0
        )

        # 新特徴量
        card_energy_type_vec = [0.0] * N_ENERGY_TYPES
        weakness_val = 0.0
        resistance_val = 0.0
        if card is not None:
            et = getattr(card, "energyType", None)
            if et is not None and 0 <= int(et) < N_ENERGY_TYPES:
                card_energy_type_vec[int(et)] = 1.0
            wk = getattr(card, "weakness", None)
            weakness_val = float(int(wk)) / N_ENERGY_TYPES if wk is not None else 0.0
            rs = getattr(card, "resistance", None)
            resistance_val = float(int(rs)) / N_ENERGY_TYPES if rs is not None else 0.0

        max_outgoing = _max_affordable_damage(card_db, attack_db, poke, None, None)
        can_attack_now = 1.0 if max_outgoing > 0 else 0.0
        appear_this_turn = 1.0 if poke.get("appearThisTurn", False) else 0.0

        return ([hp_ratio, dmg / 300.0] + energy_vec + status + card_feats
                + [incoming_max_damage / 300.0]
                + evolution_feats
                + [has_tool, has_ability, is_ex_damage_immune, has_damage_counter_immune_energy]
                + card_energy_type_vec
                + [weakness_val, resistance_val, can_attack_now, max_outgoing / 300.0, appear_this_turn])

    def get_status(player: dict) -> list[float]:
        return [
            float(player.get("poisoned", False)),
            float(player.get("burned", False)),
            float(player.get("asleep", False)),
            float(player.get("paralyzed", False)),
            float(player.get("confused", False)),
        ]

    my_active_list = my.get("active", []) or []
    my_active = my_active_list[0] if my_active_list else {}
    opp_active_list = opp.get("active", []) or []
    opp_active = opp_active_list[0] if opp_active_list else {}

    hand = my.get("hand", []) or []
    hand_ids = [get_card_id(c) for c in hand[:20]]
    hand_ids += [0] * (20 - len(hand_ids))

    bench = my.get("bench", []) or []
    bench_ids = [get_card_id(p) for p in bench[:5]]
    bench_scalars = [get_pokemon_scalar(p, threat_active=opp_active) for p in bench[:5]]
    bench_mask = [1.0] * len(bench_ids)
    pad = 5 - len(bench_ids)
    bench_ids += [0] * pad
    bench_scalars += [[0.0] * POKE_SCALAR_DIM] * pad
    bench_mask += [0.0] * pad

    opp_bench = opp.get("bench", []) or []
    opp_bench_ids = [get_card_id(p) for p in opp_bench[:5]]
    opp_bench_scalars = [get_pokemon_scalar(p, threat_active=my_active) for p in opp_bench[:5]]
    opp_bench_mask = [1.0] * len(opp_bench_ids)
    opp_bench_pad = 5 - len(opp_bench_ids)
    opp_bench_ids += [0] * opp_bench_pad
    opp_bench_scalars += [[0.0] * POKE_SCALAR_DIM] * opp_bench_pad
    opp_bench_mask += [0.0] * opp_bench_pad

    stadium = current.get("stadium", []) or []
    stadium_card = stadium[0] if isinstance(stadium, list) and stadium else stadium
    if isinstance(stadium_card, (list, tuple)):
        stadium_card = stadium_card[0] if stadium_card else None
    stadium_id = get_card_id(stadium_card)
    stadium_present = 1.0 if stadium_id else 0.0
    stadium_player_idx = (
        stadium_card.get("playerIndex") if isinstance(stadium_card, dict) else None
    )
    is_my_stadium = (
        1.0 if stadium_card is not None and stadium_player_idx == your_idx else 0.0
    )

    supporter_played = float(current.get("supporterPlayed", False))
    energy_attached_flag = float(current.get("energyAttached", False))
    retreated_flag = float(current.get("retreated", False))
    stadium_played_flag = float(current.get("stadiumPlayed", False))

    opp_discard = opp.get("discard", []) or []
    opp_discard_ids = [get_card_id(c) for c in opp_discard[:OPP_DISCARD_CAP]]
    opp_discard_mask = [1.0] * len(opp_discard_ids)
    opp_discard_pad = OPP_DISCARD_CAP - len(opp_discard_ids)
    opp_discard_ids += [0] * opp_discard_pad
    opp_discard_mask += [0.0] * opp_discard_pad

    my_ready_attackers = sum(
        _max_affordable_damage(card_db, attack_db, poke, None, None) > 0
        for poke in ([my_active] if my_active else []) + bench
    )
    opp_ready_attackers = sum(
        _max_affordable_damage(card_db, attack_db, poke, None, None) > 0
        for poke in ([opp_active] if opp_active else []) + opp_bench
    )

    # KO判定
    my_active_card = card_db.get(get_card_id(my_active)) if my_active else None
    opp_active_card = card_db.get(get_card_id(opp_active)) if opp_active else None
    opp_weakness = getattr(opp_active_card, "weakness", None) if opp_active_card else None
    opp_resistance = getattr(opp_active_card, "resistance", None) if opp_active_card else None
    my_weakness = getattr(my_active_card, "weakness", None) if my_active_card else None
    my_resistance = getattr(my_active_card, "resistance", None) if my_active_card else None
    my_active_max_dmg = _max_affordable_damage(card_db, attack_db, my_active, opp_weakness, opp_resistance)
    opp_active_max_dmg = _max_affordable_damage(card_db, attack_db, opp_active, my_weakness, my_resistance)
    opp_active_hp = float(opp_active.get("hp", 0) or 0) if opp_active else 0.0
    my_active_hp = float(my_active.get("hp", 0) or 0) if my_active else 0.0
    can_ko_opp = 1.0 if my_active_max_dmg > 0 and opp_active_hp > 0 and my_active_max_dmg >= opp_active_hp else 0.0
    opp_can_ko_me = 1.0 if opp_active_max_dmg > 0 and my_active_hp > 0 and opp_active_max_dmg >= my_active_hp else 0.0

    # 相手の捨て札のEX枚数（既にKOされたEXの数の推定）
    opp_discard_ex_count = sum(
        1 for cid in opp_discard_ids[:len(opp_discard)]
        if cid and getattr(card_db.get(cid), "ex", False)
    )

    my_prize_count = float(len(my.get("prize", []) or []))
    opp_prize_count = float(len(opp.get("prize", []) or []))

    global_scalars = [
        my_prize_count / 6.0,
        opp_prize_count / 6.0,
        float(current.get("turn", 0)) / 50.0,
        float(my.get("deckCount", 0)) / 60.0,
        float(opp.get("deckCount", 0)) / 60.0,
        float(len(my.get("bench", []) or [])) / 5.0,
        float(len(opp.get("bench", []) or [])) / 5.0,
        float(my.get("benchMax", 5)) / 5.0,
        stadium_present,
        float(my.get("handCount", 0)) / 20.0,
        float(opp.get("handCount", 0)) / 20.0,
        1.0 if current.get("firstPlayer") == your_idx else 0.0,
        float(my_ready_attackers) / 6.0,
        float(opp_ready_attackers) / 6.0,
        is_my_stadium,
        supporter_played,
        energy_attached_flag,
        retreated_flag,
        stadium_played_flag,
        # 新特徴量 (5個 → global合計24)
        can_ko_opp,
        opp_can_ko_me,
        (my_prize_count - opp_prize_count) / 6.0,  # サイド差 (正=自分が有利)
        float(current.get("turnActionCount", 0)) / 10.0,
        float(opp_discard_ex_count) / 5.0,
    ]

    return {
        "hand_ids": torch.tensor(hand_ids, dtype=torch.long, device=device).unsqueeze(0),
        "my_active_id": torch.tensor([get_card_id(my_active)], dtype=torch.long, device=device).unsqueeze(0),
        "my_active_scalar": torch.tensor(
            get_pokemon_scalar(my_active, get_status(my), opp_active),
            dtype=torch.float,
            device=device,
        ).unsqueeze(0),
        "opp_active_id": torch.tensor([get_card_id(opp_active)], dtype=torch.long, device=device).unsqueeze(0),
        "opp_active_scalar": torch.tensor(
            get_pokemon_scalar(opp_active, get_status(opp), my_active),
            dtype=torch.float,
            device=device,
        ).unsqueeze(0),
        "bench_ids": torch.tensor(bench_ids, dtype=torch.long, device=device).unsqueeze(0),
        "bench_scalar": torch.tensor(bench_scalars, dtype=torch.float, device=device).unsqueeze(0),
        "bench_mask": torch.tensor(bench_mask, dtype=torch.float, device=device).unsqueeze(0),
        "opp_bench_ids": torch.tensor(opp_bench_ids, dtype=torch.long, device=device).unsqueeze(0),
        "opp_bench_scalar": torch.tensor(
            opp_bench_scalars, dtype=torch.float, device=device
        ).unsqueeze(0),
        "opp_bench_mask": torch.tensor(opp_bench_mask, dtype=torch.float, device=device).unsqueeze(0),
        "stadium_id": torch.tensor([stadium_id], dtype=torch.long, device=device).unsqueeze(0),
        "opp_discard_ids": torch.tensor(
            opp_discard_ids, dtype=torch.long, device=device
        ).unsqueeze(0),
        "opp_discard_mask": torch.tensor(
            opp_discard_mask, dtype=torch.float, device=device
        ).unsqueeze(0),
        "global_scalars": torch.tensor(global_scalars, dtype=torch.float, device=device).unsqueeze(0),
    }


# encode_actions の追加特徴量のオフセット定義(N_OPTION_TYPES+2 = 19から開始)。
# ATTACK: 要求エネルギーのタイプ別カウント(12) + 合計コスト(1)
# ATTACH: 貼るエネルギーのタイプ(12) + 対象の現在エネルギー(12) + 対象がActiveか(1)
#         + これでワザが使用可能になるか(1) + 対象の最も近いワザまでの残り枚数(1)
_OFF_ATK_ENERGY_REQ = N_OPTION_TYPES + 2          # 19..30
_OFF_ATK_TOTAL_COST = _OFF_ATK_ENERGY_REQ + N_ENERGY_TYPES        # 31
_OFF_ATTACH_SRC_TYPE = _OFF_ATK_TOTAL_COST + 1    # 32..43
_OFF_ATTACH_TGT_ENERGY = _OFF_ATTACH_SRC_TYPE + N_ENERGY_TYPES    # 44..55
_OFF_ATTACH_TGT_ACTIVE = _OFF_ATTACH_TGT_ENERGY + N_ENERGY_TYPES  # 56
_OFF_ATTACH_UNLOCKS = _OFF_ATTACH_TGT_ACTIVE + 1  # 57
_OFF_ATTACH_REMAINING = _OFF_ATTACH_UNLOCKS + 1   # 58
_OFF_ATK_EFF_DAMAGE = _OFF_ATTACH_REMAINING + 1   # 59
_OFF_ATK_WOULD_KO = _OFF_ATK_EFF_DAMAGE + 1       # 60
OPT_ATTACH = 8
OPT_ATTACK = 13


def encode_actions(options: list, obs_dict: dict, device: str = "cpu") -> torch.Tensor:
    """行動リストをテンソルに変換。

    OptionType(int)のone-hot + ワザの要求エネルギー/ダメージ + ATTACHの
    「どのタイプのエネルギーを」「どのポケモンに(現在の保有エネルギー・このワザを使うには
    あと何枚必要か)」を構造化特徴として渡す。card_id embeddingだけに依存せず、エネルギーの
    割り振り判断(どの対象に貼るのが定石的に正しいか)を直接学習しやすくする狙い。
    """
    n = len(options)
    feats = torch.zeros(1, n, EMBED_DIM, device=device)
    if n == 0:
        return feats
    attack_db = _load_attack_db()
    card_db = _load_card_db()
    my_index = (obs_dict.get("current") or {}).get("yourIndex", 0) or 0
    my_active = _resolve_card(obs_dict, AREA_ACTIVE, 0, my_index)
    opp_active = _resolve_card(obs_dict, AREA_ACTIVE, 0, 1 - my_index)

    my_active_type = None
    if my_active is not None:
        my_active_data = card_db.get(my_active.get("id") or my_active.get("cardId"))
        energy_type = getattr(my_active_data, "energyType", None)
        if energy_type is not None:
            my_active_type = int(energy_type)

    opp_weakness = None
    opp_resistance = None
    opp_hp = 0.0
    if opp_active is not None:
        opp_active_data = card_db.get(opp_active.get("id") or opp_active.get("cardId"))
        weakness = getattr(opp_active_data, "weakness", None)
        resistance = getattr(opp_active_data, "resistance", None)
        if weakness is not None:
            opp_weakness = int(weakness)
        if resistance is not None:
            opp_resistance = int(resistance)
        opp_hp = float(opp_active.get("hp", 0) or 0)

    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        opt_type = int(opt.get("type", 14) or 14)
        if 0 <= opt_type < N_OPTION_TYPES:
            feats[0, i, opt_type] = 1.0
        feats[0, i, N_OPTION_TYPES + 1] = (opt.get("count", 0) or 0) / 8.0

        if opt_type == OPT_ATTACK:
            attack = attack_db.get(opt.get("attackId"))
            if attack is not None:
                feats[0, i, N_OPTION_TYPES] = attack.damage / 300.0
                required = [int(e) for e in attack.energies]
                for e in required:
                    if 0 <= e < N_ENERGY_TYPES:
                        feats[0, i, _OFF_ATK_ENERGY_REQ + e] += 1.0 / 4.0
                feats[0, i, _OFF_ATK_TOTAL_COST] = len(required) / 4.0
                effective_damage = float(attack.damage)
                if my_active_type is not None and opp_weakness is not None:
                    if my_active_type == opp_weakness:
                        effective_damage *= 2
                if my_active_type is not None and opp_resistance is not None:
                    if my_active_type == opp_resistance:
                        effective_damage = max(0.0, effective_damage - 30)
                feats[0, i, _OFF_ATK_EFF_DAMAGE] = effective_damage / 300.0
                if opp_hp > 0 and effective_damage >= opp_hp:
                    feats[0, i, _OFF_ATK_WOULD_KO] = 1.0

        elif opt_type == OPT_ATTACH:
            src_card = _resolve_card(obs_dict, opt.get("area"), opt.get("index"), my_index)
            src_type = None
            if src_card is not None:
                src_data = card_db.get(src_card.get("id") or src_card.get("cardId"))
                if src_data is not None:
                    src_type = int(src_data.energyType)
                    if 0 <= src_type < N_ENERGY_TYPES:
                        feats[0, i, _OFF_ATTACH_SRC_TYPE + src_type] = 1.0

            in_play_area = opt.get("inPlayArea")
            target = _resolve_card(obs_dict, in_play_area, opt.get("inPlayIndex"), my_index)
            if target is not None:
                tgt_energy = _energy_count_vec(target.get("energies"))
                for j, v in enumerate(tgt_energy):
                    feats[0, i, _OFF_ATTACH_TGT_ENERGY + j] = v / 4.0
                feats[0, i, _OFF_ATTACH_TGT_ACTIVE] = 1.0 if in_play_area == AREA_ACTIVE else 0.0
                unlocks, remaining = _attach_target_features(card_db, attack_db, target, src_type)
                feats[0, i, _OFF_ATTACH_UNLOCKS] = unlocks
                feats[0, i, _OFF_ATTACH_REMAINING] = remaining / 4.0

    return feats


class RLAgent:
    """PPO学習済みエージェント（推論モード）"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.net = PTCGNet()
        if model_path:
            ckpt = torch.load(model_path, map_location=device)
            self.net.load_state_dict(ckpt["model"], strict=False)
        self.net.eval()

    def __call__(self, obs_dict: dict) -> list[int]:
        select = obs_dict["select"]
        options = select["option"] or []
        n = len(options)

        if n == 0:
            return []

        max_count = select.get("maxCount", 1) or 1
        min_count = select.get("minCount", 1) or 1
        k = max(min_count, min(max_count, n))

        with torch.no_grad():
            state = encode_state(obs_dict, self.device)
            action_feats = encode_actions(options, obs_dict, self.device)
            logits, _ = self.net(state, action_feats)
            probs = F.softmax(logits[0], dim=-1)
            selected = torch.topk(probs, k).indices.tolist()

        return selected
