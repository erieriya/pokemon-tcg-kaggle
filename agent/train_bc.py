"""
Behavior Cloning事前学習スクリプト

dragapult_agent_v2(search_begin統合済みのヒューリスティック)の意思決定を教師データとして、
agent/rl_agent.py の PTCGNet を事前学習する。

狙い: ゼロ(ランダム初期化)からPPO自己対戦を始める代わりに、「定石をある程度再現できる」
状態から学習を始められるようにする。学習後のチェックポイントは
`train_ppo.py --resume <このスクリプトのoutパス>` でPPO自己対戦の初期値として使える
(checkpoint形式はPPOTrainer.save/loadと互換: {"model", "optimizer", "episode"})。

使い方:
  cd agent && uv run python train_bc.py --games 200 --vs_opponent_games 20 --epochs 3 --out models/bc_pretrain.pt
"""

import argparse
import os
import random
import sys

import torch
import torch.optim as optim

CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_start, battle_select, battle_finish

from rl_agent import PTCGNet, encode_state, encode_actions
from train_ppo import read_deck, _sequential_log_prob, FIXED_OPPONENTS

import dragapult_agent_v2


def collect_self_play(deck: list[int], n_games: int, max_steps: int = 400) -> list[tuple[dict, list[int]]]:
    """dragapult_agent_v2同士の自己対戦(deck.csvのミラー)から(obs_dict, action)を収集する。"""
    samples = []
    for _ in range(n_games):
        obs_dict, _ = battle_start(deck, deck)
        steps = 0
        while steps < max_steps:
            state = obs_dict.get("current") or {}
            if state.get("result", -1) != -1:
                break
            sel = obs_dict.get("select")
            if sel is None:
                break
            options = sel.get("option") or []
            action = dragapult_agent_v2.agent(obs_dict)
            if options and action:
                samples.append((obs_dict, list(action)))
            try:
                obs_dict = battle_select(action)
            except IndexError:
                break
            steps += 1
        battle_finish()
    return samples


def collect_vs_opponents(
    my_deck: list[int], n_games_per_opponent: int, max_steps: int = 400
) -> list[tuple[dict, list[int]]]:
    """train_ppo.pyのFIXED_OPPONENTSと対戦させ、自分側(dragapult_agent_v2)の決定だけを収集する。"""
    samples = []
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    for name, (opp_fn, opp_deck_name) in FIXED_OPPONENTS.items():
        opp_deck = my_deck if opp_deck_name is None else read_deck(os.path.join(agent_dir, opp_deck_name))
        for _ in range(n_games_per_opponent):
            net_idx = random.randint(0, 1)
            deck0 = my_deck if net_idx == 0 else opp_deck
            deck1 = opp_deck if net_idx == 0 else my_deck
            obs_dict, _ = battle_start(deck0, deck1)
            steps = 0
            while steps < max_steps:
                state = obs_dict.get("current") or {}
                if state.get("result", -1) != -1:
                    break
                sel = obs_dict.get("select")
                if sel is None:
                    break
                player_idx = state.get("yourIndex", 0)
                options = sel.get("option") or []
                if player_idx == net_idx:
                    action = dragapult_agent_v2.agent(obs_dict)
                    if options and action:
                        samples.append((obs_dict, list(action)))
                else:
                    action = opp_fn(obs_dict) if options else []
                try:
                    obs_dict = battle_select(action)
                except IndexError:
                    break
                steps += 1
            battle_finish()
        print(f"[BC] collected vs {name}: running total continues...")
    return samples


def train(
    samples: list[tuple[dict, list[int]]], epochs: int, lr: float, device: str
) -> tuple[PTCGNet, torch.optim.Optimizer]:
    net = PTCGNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        count = 0
        for obs_dict, action in samples:
            sel = obs_dict.get("select") or {}
            options = sel.get("option") or []
            valid_action = [i for i in action if 0 <= i < len(options)]
            if not options or not valid_action:
                continue
            state = encode_state(obs_dict, device)
            action_feats = encode_actions(options, obs_dict, device)
            logits, _value = net(state, action_feats)
            log_prob, _entropy = _sequential_log_prob(logits[0], valid_action)
            loss = -log_prob
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / max(1, count)
        print(f"[BC] epoch={epoch} avg_loss={avg_loss:.4f} samples_used={count}")
    return net, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200, help="自己対戦(ミラー)の試合数")
    parser.add_argument("--vs_opponent_games", type=int, default=20, help="対戦相手プール1体あたりの試合数")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/bc_pretrain.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    my_deck = read_deck("deck.csv")

    print(f"[BC] device={args.device} search_enabled={dragapult_agent_v2.SEARCH_ENABLED}")
    print(f"[BC] collecting {args.games} self-play games...")
    samples = collect_self_play(my_deck, args.games)
    print(f"[BC] collected {len(samples)} self-play samples")

    if args.vs_opponent_games > 0:
        print(f"[BC] collecting {args.vs_opponent_games} games per opponent...")
        samples += collect_vs_opponents(my_deck, args.vs_opponent_games)
        print(f"[BC] total samples after opponent pool: {len(samples)}")

    net, optimizer = train(samples, args.epochs, args.lr, args.device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "episode": 0},
        args.out,
    )
    print(f"[BC] saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
