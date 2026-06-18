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
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# カードIDの最大値（暫定: EN_Card_Data.csvを確認後に更新）
MAX_CARD_ID = 3000
EMBED_DIM = 128
STATE_DIM = 256
HIDDEN_DIM = 256
N_ENERGY_TYPES = 12  # cg.api.EnergyType: COLORLESS(0)〜TEAM_ROCKET(11)
N_OPTION_TYPES = 17  # cg.api.OptionType: NUMBER(0)〜SPECIAL_CONDITION(16)


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
        # HP比率 + ダメージ量 + エネルギー数(N_ENERGY_TYPES) + ステータス異常5種 + カード静的特徴4
        scalar_dim = 1 + 1 + N_ENERGY_TYPES + 5 + 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + scalar_dim, HIDDEN_DIM),
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
        # hand_vec(EMBED_DIM) + my_active(HIDDEN_DIM) + opp_active(HIDDEN_DIM) + bench(EMBED_DIM) + scalars(8)
        concat_dim = EMBED_DIM + HIDDEN_DIM + HIDDEN_DIM + EMBED_DIM + 8
        self.global_proj = nn.Linear(concat_dim, STATE_DIM)

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

        # ベンチ（平均プーリング）
        bench_ids = state["bench_ids"]
        bench_embs = self.card_emb(bench_ids)
        bench_vec = bench_embs.mean(dim=1)

        # グローバル情報（サイド枚数、ターン数等）
        global_scalars = state["global_scalars"]

        combined = torch.cat([
            hand_vec,
            my_active_vec,
            opp_active_vec,
            bench_vec,
            global_scalars,
        ], dim=-1)
        return F.relu(self.global_proj(combined))


class PTCGNet(nn.Module):
    """PolicyとValueを出力するネットワーク"""

    def __init__(self, max_actions: int = 100):
        super().__init__()
        self.state_enc = StateEncoder()
        self.card_emb = CardEmbedding()

        # action_features (EMBED_DIM) を STATE_DIM に投影
        self.action_proj = nn.Linear(EMBED_DIM, STATE_DIM)
        self.policy_attn = nn.MultiheadAttention(
            embed_dim=STATE_DIM, num_heads=4, batch_first=True
        )
        self.policy_head = nn.Linear(STATE_DIM, 1)
        self.value_head = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        state: dict,
        action_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: encode_state()の出力
            action_features: (batch, n_actions, EMBED_DIM) 各行動の特徴量
        Returns:
            logits: (batch, n_actions)
            value: (batch, 1)
        """
        state_vec = self.state_enc(state)

        # action_features を STATE_DIM に投影
        action_proj = self.action_proj(action_features)  # (B, n_actions, STATE_DIM)

        q = state_vec.unsqueeze(1)  # (B, 1, STATE_DIM)
        attn_out, _ = self.policy_attn(q, action_proj, action_proj)
        logits = self.policy_head(attn_out + action_proj).squeeze(-1)  # (B, n_actions)

        value = self.value_head(state_vec)
        return logits, value


_CARD_DB: dict[int, object] | None = None
_ATTACK_DB: dict[int, object] | None = None


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

    def get_card_id(card) -> int:
        if card is None:
            return 0
        if isinstance(card, dict):
            return card.get("id", card.get("cardId", 0)) or 0
        return int(card)

    def get_pokemon_scalar(poke: dict, status: list[float] | None = None) -> list[float]:
        if not poke:
            return [0.0] * (2 + N_ENERGY_TYPES + 5 + 4)
        cur_hp = float(poke.get("hp", 0) or 0)
        max_hp = float(poke.get("maxHp", 0) or cur_hp or 100)
        dmg = max(0.0, max_hp - cur_hp)
        hp_ratio = cur_hp / max_hp if max_hp > 0 else 0.0

        energy_vec = [0.0] * N_ENERGY_TYPES
        for e in poke.get("energies", []) or []:
            idx = int(e)
            if 0 <= idx < N_ENERGY_TYPES:
                energy_vec[idx] += 1.0

        status = status or [0.0] * 5
        card_feats = _card_static_feats(get_card_id(poke))

        return [hp_ratio, dmg / 300.0] + energy_vec + status + card_feats

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
    bench_ids += [0] * (5 - len(bench_ids))

    global_scalars = [
        float(len(my.get("prize", []) or [])),
        float(len(opp.get("prize", []) or [])),
        float(current.get("turn", 0)),
        float(my.get("deckCount", 0)) / 60.0,
        float(opp.get("deckCount", 0)) / 60.0,
        float(len(my.get("bench", []) or [])),
        float(len(opp.get("bench", []) or [])),
        float(my.get("benchMax", 5)),
    ]

    return {
        "hand_ids": torch.tensor(hand_ids, dtype=torch.long, device=device).unsqueeze(0),
        "my_active_id": torch.tensor([get_card_id(my_active)], dtype=torch.long, device=device).unsqueeze(0),
        "my_active_scalar": torch.tensor(
            get_pokemon_scalar(my_active, get_status(my)), dtype=torch.float, device=device
        ).unsqueeze(0),
        "opp_active_id": torch.tensor([get_card_id(opp_active)], dtype=torch.long, device=device).unsqueeze(0),
        "opp_active_scalar": torch.tensor(
            get_pokemon_scalar(opp_active, get_status(opp)), dtype=torch.float, device=device
        ).unsqueeze(0),
        "bench_ids": torch.tensor(bench_ids, dtype=torch.long, device=device).unsqueeze(0),
        "global_scalars": torch.tensor(global_scalars, dtype=torch.float, device=device).unsqueeze(0),
    }


def encode_actions(options: list, device: str = "cpu") -> torch.Tensor:
    """行動リストをテンソルに変換。OptionType(int)のone-hot + ワザダメージ等のスカラー特徴。"""
    n = len(options)
    feats = torch.zeros(1, n, EMBED_DIM, device=device)
    if n == 0:
        return feats
    attack_db = _load_attack_db()
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        opt_type = int(opt.get("type", 14) or 14)
        if 0 <= opt_type < N_OPTION_TYPES:
            feats[0, i, opt_type] = 1.0
        attack = attack_db.get(opt.get("attackId"))
        feats[0, i, N_OPTION_TYPES] = (attack.damage if attack else 0) / 300.0
        feats[0, i, N_OPTION_TYPES + 1] = (opt.get("count", 0) or 0) / 8.0
    return feats


class RLAgent:
    """PPO学習済みエージェント（推論モード）"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.net = PTCGNet()
        if model_path:
            ckpt = torch.load(model_path, map_location=device)
            self.net.load_state_dict(ckpt["model"])
        self.net.eval()

    def __call__(self, obs_dict: dict) -> list[int]:
        select = obs_dict["select"]
        options = select["option"]
        max_count = select["maxCount"]

        if not options:
            return []

        with torch.no_grad():
            state = encode_state(obs_dict, self.device)
            action_feats = encode_actions(options, self.device)
            logits, _ = self.net(state, action_feats)
            probs = F.softmax(logits[0], dim=-1)
            k = min(max_count, len(options))
            selected = torch.topk(probs, k).indices.tolist()

        return selected
