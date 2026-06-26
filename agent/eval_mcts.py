"""
MCTSチェックポイントを固定対戦相手プールに対して評価するスクリプト。

使い方:
  cd agent && uv run python eval_mcts.py --checkpoint models/mcts_loop_v2_gen5.pt \
      --games 10 --simulations 64 --candidates 4 --max_steps 300
"""

import argparse
import os
import random
import sys

import torch

CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

import mcts  # noqa: E402
from rl_agent import PTCGNet  # noqa: E402
from train_ppo import FIXED_OPPONENTS, _sanitize_opponent_action, read_deck  # noqa: E402


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fallback_action(sel: dict, n: int) -> list[int]:
    """train_mcts.pyのplay_one_gameと同じMCTS失敗時フォールバック。"""
    if n == 0:
        return []
    k = max(sel.get("minCount", 1) or 1, min(sel.get("maxCount", 1) or 1, n))
    return list(range(k))


def load_checkpoint(path: str, device: str) -> PTCGNet:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    net = PTCGNet()
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[WARN] loaded {path} with strict=False "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
    net.to(device)
    net.eval()
    return net


def play_mcts_eval_game(
    net: PTCGNet,
    my_deck: list[int],
    opponent_fn,
    opponent_deck: list[int],
    device: str,
    n_simulations: int,
    num_candidates: int,
    max_steps: int,
) -> tuple[int, int]:
    """MCTS探索込みのnetと固定相手を1戦対戦させ、(result, net_idx)を返す。"""
    net_idx = random.randint(0, 1)
    deck0 = my_deck if net_idx == 0 else opponent_deck
    deck1 = opponent_deck if net_idx == 0 else my_deck
    obs_dict, _ = battle_start(deck0, deck1)
    step = 0

    try:
        while step < max_steps:
            state = obs_dict.get("current") or {}
            result = state.get("result", -1)
            sel = obs_dict.get("select")
            if result != -1 or sel is None:
                break

            player_idx = state.get("yourIndex", 0)
            options = sel.get("option") or []
            n = len(options)

            if player_idx == net_idx:
                if n == 0:
                    action = []
                else:
                    with torch.no_grad():
                        action, _policy_target = mcts.search_policy(
                            obs_dict,
                            net,
                            my_deck,
                            device=device,
                            n_simulations=n_simulations,
                            num_candidates=num_candidates,
                        )
                    if action is None:
                        action = _fallback_action(sel, n)
            else:
                action = opponent_fn(obs_dict) if n else []
                action = _sanitize_opponent_action(action, sel, n)

            try:
                obs_dict = battle_select(action)
            except IndexError:
                break
            step += 1

        return (obs_dict.get("current") or {}).get("result", -1), net_idx
    finally:
        battle_finish()


def evaluate_vs_opponent(
    net: PTCGNet,
    my_deck: list[int],
    opponent_fn,
    opponent_deck: list[int],
    device: str,
    games: int,
    n_simulations: int,
    num_candidates: int,
    max_steps: int,
) -> dict[str, int | float]:
    wins = losses = draws = 0
    for _ in range(games):
        result, net_idx = play_mcts_eval_game(
            net,
            my_deck,
            opponent_fn,
            opponent_deck,
            device,
            n_simulations,
            num_candidates,
            max_steps,
        )
        if result == net_idx:
            wins += 1
        elif result in (0, 1):
            losses += 1
        else:
            draws += 1

    win_rate = wins / games if games else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }


def parse_opponents(opponents_arg: str | None) -> list[str]:
    if opponents_arg is None:
        return list(FIXED_OPPONENTS.keys())
    names = [name.strip() for name in opponents_arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in FIXED_OPPONENTS]
    if unknown:
        valid = ", ".join(FIXED_OPPONENTS.keys())
        raise ValueError(f"unknown opponents: {', '.join(unknown)} (valid: {valid})")
    return names


def build_fixed_opponents(names: list[str], my_deck: list[int]):
    opponents = []
    for name in names:
        opponent_fn, deck_filename = FIXED_OPPONENTS[name]
        opponent_deck = (
            my_deck
            if deck_filename is None
            else read_deck(os.path.join(AGENT_DIR, deck_filename))
        )
        opponents.append((name, opponent_fn, opponent_deck))
    return opponents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--opponents", type=str, default=None)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    my_deck = read_deck(os.path.join(AGENT_DIR, "deck.csv"))
    opponent_names = parse_opponents(args.opponents)
    fixed_opponents = build_fixed_opponents(opponent_names, my_deck)
    summary: list[tuple[str, float]] = []

    for checkpoint in args.checkpoint:
        print(f"\n=== {checkpoint} ===")
        net = load_checkpoint(checkpoint, args.device)
        win_rates = []
        for name, opponent_fn, opponent_deck in fixed_opponents:
            stats = evaluate_vs_opponent(
                net,
                my_deck,
                opponent_fn,
                opponent_deck,
                args.device,
                args.games,
                args.simulations,
                args.candidates,
                args.max_steps,
            )
            win_rates.append(float(stats["win_rate"]))
            print(
                f"{name:12s} "
                f"win_rate={stats['win_rate'] * 100:6.1f}% "
                f"games={stats['games']:3d} "
                f"wins={stats['wins']:3d} "
                f"losses={stats['losses']:3d} "
                f"draws={stats['draws']:3d}"
            )
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        summary.append((checkpoint, avg_win_rate))

    print("\n=== Summary ===")
    for checkpoint, avg_win_rate in summary:
        print(f"{checkpoint}: average_win_rate={avg_win_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
