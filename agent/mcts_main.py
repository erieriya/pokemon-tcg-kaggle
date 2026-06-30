"""
PTCG AI Battle Challenge - PTCGNet + PUCT MCTS エージェント 提出用エントリポイント

submit_mcts.sh がこのファイルを main.py としてリネームし、以下と一緒にtar.gz化する:
  rl_agent.py, mcts.py, train_ppo.py, model_final.pt, deck.csv, cg/

MCTS が利用できない局面 (search_begin_input=None) は PTCGNet の Greedy 推論にフォールバック。
"""

import os
import torch
import torch.nn.functional as F

# Kaggle実行環境では main.py を exec() で実行するため __file__ が未定義になる。
try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = "/kaggle_simulations/agent"

import sys
sys.path.insert(0, _AGENT_DIR)

from rl_agent import PTCGNet, encode_state, encode_actions
import mcts as mcts_mod

# ---------- 設定 ----------
N_SIMULATIONS = 64
NUM_CANDIDATES = 6
MIN_CANDIDATES = 1
DYNAMIC_CANDIDATES = True
# --------------------------


def _resolve(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    p = os.path.join(_AGENT_DIR, filename)
    if os.path.exists(p):
        return p
    return "/kaggle_simulations/agent/" + filename


def _read_deck() -> list[int]:
    with open(_resolve("deck.csv")) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


_net = None
_deck: list[int] | None = None


def _get_net() -> PTCGNet:
    global _net
    if _net is None:
        _net = PTCGNet()
        model_path = _resolve("model_final.pt")
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            _net.load_state_dict(ckpt["model"], strict=False)
        _net.eval()
    return _net


def _get_deck() -> list[int]:
    global _deck
    if _deck is None:
        _deck = _read_deck()
    return _deck


def _greedy_action(obs_dict: dict) -> list[int]:
    """MCTS が使えない場合の PTCGNet Greedy 推論"""
    select = obs_dict.get("select") or {}
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []
    max_count = select.get("maxCount", 1) or 1
    min_count = select.get("minCount", 1) or 1
    k = max(min_count, min(max_count, n))
    net = _get_net()
    with torch.no_grad():
        state = encode_state(obs_dict, "cpu")
        action_feats = encode_actions(options, obs_dict, "cpu")
        logits, _ = net(state, action_feats)
        probs = F.softmax(logits[0, :n], dim=-1)
        selected = torch.topk(probs, k).indices.tolist()
    return selected


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return _read_deck()

    net = _get_net()
    deck = _get_deck()

    best_action, _ = mcts_mod.search_policy(
        obs_dict,
        net,
        deck,
        device="cpu",
        n_simulations=N_SIMULATIONS,
        num_candidates=NUM_CANDIDATES,
        min_candidates=MIN_CANDIDATES,
        dynamic_candidates=DYNAMIC_CANDIDATES,
        add_noise_root=False,  # 推論時はノイズ無し
    )

    if best_action is None:
        return _greedy_action(obs_dict)

    return best_action
