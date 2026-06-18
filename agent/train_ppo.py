"""
PTCG AI Battle Challenge - PPO 自己対戦学習スクリプト

使い方:
  uv run python agent/train_ppo.py --episodes 10000 --save_interval 500

cg/ ライブラリは agent/deck.csv と同じ data/sample_submission/cg を自動で読み込む。
"""

import argparse
import os
import random
import sys
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_start, battle_select, battle_finish

from rl_agent import PTCGNet, encode_state, encode_actions


def _sequential_sample(logits: torch.Tensor, k: int) -> tuple[list[int], torch.Tensor]:
    """重複なしでk個選ぶ（選んだ選択肢のlogitを-infにして再softmax）。
    minCount/maxCountが1より大きいselect（ベンチに複数出す等）に対応するため。"""
    logits = logits.clone()
    chosen: list[int] = []
    total_log_prob = torch.zeros((), device=logits.device)
    for _ in range(k):
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        total_log_prob = total_log_prob + dist.log_prob(idx)
        chosen.append(idx.item())
        logits[idx] = -1e9
    return chosen, total_log_prob


def _sequential_log_prob(logits: torch.Tensor, action_idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """指定した順序でindexを選んだ場合の対数確率合計とエントロピー合計（PPO更新時に使用）。"""
    logits = logits.clone()
    total_log_prob = torch.zeros((), device=logits.device)
    total_entropy = torch.zeros((), device=logits.device)
    for idx in action_idxs:
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        total_log_prob = total_log_prob + dist.log_prob(torch.tensor(idx, device=logits.device))
        total_entropy = total_entropy + dist.entropy()
        logits[idx] = -1e9
    return total_log_prob, total_entropy


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


# ============================================================
# Self-play環境（cgエンジンを直接ラップ）
# ============================================================

class PTCGSelfPlayEnv:
    """同じpolicyが両プレイヤーを操作する自己対戦環境。

    deck.csv固定デッキでのミラーマッチ（research.md Phase3でデッキ可変化はTODO）。
    """

    def __init__(self, deck_path: str):
        self.deck = read_deck(deck_path)

    def play_episode(self, trainer: "PPOTrainer", max_steps: int = 2000) -> dict:
        """1試合実行し、両プレイヤー視点の遷移をtrainerのバッファへ積む。

        各プレイヤーは自分の手番の連続をそれぞれ独立した軌跡として記録し、
        試合終了時にそのプレイヤー自身の最後の手番にだけ勝敗報酬を入れる
        （手番が交互に来るため、相手の行動を挟んでも各プレイヤーのGAEは
        そのプレイヤー自身の軌跡内だけで計算される）。
        """
        obs_dict, _ = battle_start(self.deck, self.deck)
        last_idx = {0: None, 1: None}
        step = 0
        result = -1

        while step < max_steps:
            state = obs_dict.get("current") or {}
            result = state.get("result", -1)
            sel = obs_dict.get("select")
            if result != -1 or sel is None:
                break

            player_idx = state.get("yourIndex", 0)
            action, log_prob, value = trainer.select_action(obs_dict)
            trainer.store(player_idx, obs_dict, action, log_prob, value, reward=0.0, done=False)
            last_idx[player_idx] = len(trainer.buffers[player_idx]) - 1

            obs_dict = battle_select(action)
            step += 1

        result = (obs_dict.get("current") or {}).get("result", -1)
        turn = (obs_dict.get("current") or {}).get("turn", 0)
        battle_finish()

        for player_idx, idx in last_idx.items():
            if idx is None:
                continue
            if result == player_idx:
                reward = 1.0
            elif result in (0, 1):
                reward = -1.0
            else:
                reward = 0.0
            buf = trainer.buffers[player_idx][idx]
            buf["reward"] = reward
            buf["done"] = True

        return {"result": result, "steps": step, "turns": turn}


def play_eval_game(net: "PTCGNet", deck: list[int], opponent: str, device: str, max_steps: int = 2000) -> int:
    """学習中のpolicy(player0) vs ランダム/ヒューリスティック(player1)を1戦して結果を返す。"""
    obs_dict, _ = battle_start(deck, deck)
    step = 0
    result = -1
    while step < max_steps:
        state = obs_dict.get("current") or {}
        result = state.get("result", -1)
        sel = obs_dict.get("select")
        if result != -1 or sel is None:
            break

        player_idx = state.get("yourIndex", 0)
        options = sel.get("option") or []
        n = len(options)
        max_c = sel.get("maxCount", 1)
        min_c = sel.get("minCount", 1)
        k = max(min_c, min(max_c, n))

        if player_idx == 0:
            if n == 0:
                action = []
            else:
                with torch.no_grad():
                    s = encode_state(obs_dict, device)
                    a = encode_actions(options, device)
                    logits, _ = net(s, a)
                    probs = F.softmax(logits[0], dim=-1)
                action = torch.topk(probs, k).indices.tolist()
        else:
            action = random.sample(range(n), k) if n else []

        obs_dict = battle_select(action)
        step += 1

    result = (obs_dict.get("current") or {}).get("result", -1)
    battle_finish()
    return result


def evaluate(net: "PTCGNet", deck: list[int], device: str, n_games: int = 10) -> float:
    """ランダムエージェント相手の勝率を返す（学習が進んでいるかの確認用）。"""
    wins = 0
    for _ in range(n_games):
        result = play_eval_game(net, deck, "random", device)
        if result == 0:
            wins += 1
    return wins / n_games


# ============================================================
# PPO実装
# ============================================================

class PPOTrainer:
    def __init__(
        self,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        n_epochs: int = 4,
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs

        self.net = PTCGNet().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # プレイヤーごとに別の軌跡として保持する（手番が交互に来るため、
        # 1本の配列に混ぜるとGAEのbootstrap先が相手の状態価値になってしまう）。
        self.buffers: dict[int, list[dict]] = {0: [], 1: []}

    def n_stored(self) -> int:
        return len(self.buffers[0]) + len(self.buffers[1])

    def select_action(self, obs_dict: dict) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        select = obs_dict["select"]
        options = select["option"] or []
        n = len(options)

        if n == 0:
            return [], torch.tensor(0.0), torch.tensor(0.0)

        max_count = select.get("maxCount", 1) or 1
        min_count = select.get("minCount", 1) or 1
        k = max(min_count, min(max_count, n))

        state = encode_state(obs_dict, self.device)
        action_feats = encode_actions(options, self.device)

        with torch.no_grad():
            logits, value = self.net(state, action_feats)

        chosen, log_prob = _sequential_sample(logits[0], k)
        return chosen, log_prob, value.squeeze()

    def store(self, player_idx: int, obs: dict, action: list[int], log_prob: torch.Tensor,
              value: torch.Tensor, reward: float, done: bool):
        self.buffers[player_idx].append({
            "obs": obs,
            "action": action,
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": reward,
            "done": done,
        })

    def _compute_gae(self, buf: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """1プレイヤー分の軌跡に対するGeneralized Advantage Estimation"""
        rewards = [b["reward"] for b in buf]
        values = torch.stack([b["value"] for b in buf])
        dones = [b["done"] for b in buf]

        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.gamma * next_value - values[t].item()
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t].item()
            next_value = values[t].item()

        return advantages, returns

    def update(self) -> dict:
        entries: list[dict] = []
        adv_parts, ret_parts = [], []
        for player_idx in (0, 1):
            buf = self.buffers[player_idx]
            if not buf:
                continue
            adv, ret = self._compute_gae(buf)
            entries.extend(buf)
            adv_parts.append(adv)
            ret_parts.append(ret)

        if not entries:
            return {}

        advantages = torch.cat(adv_parts)
        returns = torch.cat(ret_parts)
        old_log_probs = torch.stack([b["log_prob"] for b in entries])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = policy_loss = value_loss = entropy_loss = 0.0
        n = len(entries)

        for _ in range(self.n_epochs):
            for i, buf in enumerate(entries):
                options = buf["obs"]["select"]["option"]
                if not options or not buf["action"]:
                    continue
                state = encode_state(buf["obs"], self.device)
                action_feats = encode_actions(options, self.device)
                logits, value = self.net(state, action_feats)
                log_prob, entropy = _sequential_log_prob(logits[0], buf["action"])

                ratio = torch.exp(log_prob - old_log_probs[i])
                adv = advantages[i].to(self.device)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                p_loss = -torch.min(surr1, surr2)
                v_loss = F.mse_loss(value.squeeze(), returns[i].to(self.device))
                e_loss = -self.entropy_coef * entropy

                loss = p_loss + self.value_coef * v_loss + e_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += p_loss.item()
                value_loss += v_loss.item()
                entropy_loss += e_loss.item()

        self.buffers = {0: [], 1: []}
        return {
            "loss": total_loss / n,
            "policy_loss": policy_loss / n,
            "value_loss": value_loss / n,
            "entropy_loss": entropy_loss / n,
        }

    def save(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": episode,
        }, path)
        print(f"[SAVE] episode={episode} → {path}")

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("episode", 0)


# ============================================================
# 学習ループ
# ============================================================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    deck_path = os.path.join(os.path.dirname(__file__), "deck.csv")
    trainer = PPOTrainer(device=device)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        start_ep = trainer.load(args.resume)
        print(f"Resumed from episode {start_ep}")

    env = PTCGSelfPlayEnv(deck_path)
    turn_history = deque(maxlen=100)
    draw_history = deque(maxlen=100)
    UPDATE_EVERY = args.update_every

    for ep in range(start_ep, args.episodes):
        info = env.play_episode(trainer, max_steps=args.max_steps)
        turn_history.append(info["turns"])
        draw_history.append(1.0 if info["result"] not in (0, 1) else 0.0)

        if trainer.n_stored() >= UPDATE_EVERY:
            trainer.update()

        if ep % args.log_interval == 0:
            avg_turns = sum(turn_history) / len(turn_history) if turn_history else 0
            draw_rate = sum(draw_history) / len(draw_history) if draw_history else 0
            print(f"[EP {ep:5d}] avg_turns(100)={avg_turns:.1f} draw_rate(100)={draw_rate:.1%}")

        if args.eval_interval and ep > 0 and ep % args.eval_interval == 0:
            win_rate = evaluate(trainer.net, env.deck, device, n_games=args.eval_games)
            print(f"[EP {ep:5d}] vs random win_rate({args.eval_games})={win_rate:.1%}")

        if (ep + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_ep{ep+1}.pt")
            trainer.save(save_path, ep + 1)

    final_path = os.path.join(args.save_dir, "model_final.pt")
    trainer.save(final_path, args.episodes)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=2000, help="1試合あたりの最大ステップ数")
    parser.add_argument("--update_every", type=int, default=256, help="このステップ数蓄積したらPPO更新")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="../models")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200, help="0で評価無効")
    parser.add_argument("--eval_games", type=int, default=10)
    args = parser.parse_args()
    train(args)
