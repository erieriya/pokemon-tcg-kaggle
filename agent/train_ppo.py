"""
PTCG AI Battle Challenge - PPO 自己対戦学習スクリプト

使い方:
  python train_ppo.py --episodes 10000 --save_interval 500

NOTE: cg/ ライブラリへのパスを通してから実行すること:
  export PYTHONPATH=/path/to/data/sample_submission:$PYTHONPATH
"""

import argparse
import os
import sys
import random
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# cg/ ライブラリを追加（実際のパスに変更すること）
CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

try:
    import cg
    HAS_CG = True
except ImportError:
    HAS_CG = False
    print("[WARNING] cg library not found. Using mock environment for testing.")

from rl_agent import PTCGNet, encode_state, encode_actions, EMBED_DIM


# ============================================================
# Mock環境（cgなしでロジックをテストするため）
# ============================================================

class MockEnv:
    """cg環境のモック（実際のAPIが使えない場合のテスト用）"""

    def reset(self) -> dict:
        return self._random_obs()

    def step(self, action: list[int]) -> tuple[dict, float, bool]:
        done = random.random() < 0.05
        reward = 1.0 if done and random.random() < 0.5 else 0.0
        return self._random_obs(), reward, done

    def _random_obs(self) -> dict:
        n_options = random.randint(1, 10)
        return {
            "logs": [],
            "current": {
                "players": [
                    {
                        "active": [{"id": random.randint(1, 100), "hp": 200, "damageCounters": random.randint(0, 180)}],
                        "bench": [{"id": random.randint(1, 100)} for _ in range(random.randint(0, 3))],
                        "hand": [{"id": random.randint(1, 100)} for _ in range(random.randint(4, 7))],
                        "prize": [None] * random.randint(1, 6),
                        "deckCount": random.randint(10, 50),
                    },
                    {
                        "active": [{"id": random.randint(1, 100), "hp": 200, "damageCounters": random.randint(0, 180)}],
                        "bench": [],
                        "handCount": 5,
                        "prize": [None] * random.randint(1, 6),
                        "deckCount": random.randint(10, 50),
                    },
                ],
                "turn": random.randint(1, 20),
            },
            "select": {
                "option": [{"type": random.choice(["attack", "pass", "attach_energy", "play_basic"])} for _ in range(n_options)],
                "maxCount": 1,
            },
        }


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
        batch_size: int = 64,
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.net = PTCGNet().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.rollout_buffer: list[dict] = []

    def select_action(self, obs_dict: dict) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        select = obs_dict["select"]
        options = select["option"]
        max_count = select["maxCount"]

        if not options:
            return [], torch.tensor(0.0), torch.tensor(0.0)

        state = encode_state(obs_dict, self.device)
        action_feats = encode_actions(options, self.device)

        with torch.no_grad():
            logits, value = self.net(state, action_feats)
            probs = F.softmax(logits[0], dim=-1)

        dist = torch.distributions.Categorical(probs)
        k = min(max_count, len(options))
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        return [action_idx.item()], log_prob, value.squeeze()

    def store(self, obs: dict, action: list[int], log_prob: torch.Tensor,
              value: torch.Tensor, reward: float, done: bool):
        self.rollout_buffer.append({
            "obs": obs,
            "action": action,
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": reward,
            "done": done,
        })

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation"""
        rewards = [b["reward"] for b in self.rollout_buffer]
        values = torch.stack([b["value"] for b in self.rollout_buffer])
        dones = [b["done"] for b in self.rollout_buffer]

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

    def update(self):
        if not self.rollout_buffer:
            return {}

        advantages, returns = self.compute_gae()
        old_log_probs = torch.stack([b["log_prob"] for b in self.rollout_buffer])

        # 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = policy_loss = value_loss = entropy_loss = 0.0

        for _ in range(self.n_epochs):
            for i, buf in enumerate(self.rollout_buffer):
                state = encode_state(buf["obs"], self.device)
                options = buf["obs"]["select"]["option"]
                if not options or not buf["action"]:
                    continue
                action_feats = encode_actions(options, self.device)
                logits, value = self.net(state, action_feats)
                probs = F.softmax(logits[0], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_tensor = torch.tensor(buf["action"][0], dtype=torch.long, device=self.device)
                log_prob = dist.log_prob(action_tensor)
                entropy = dist.entropy()

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

        self.rollout_buffer.clear()
        n = max(len(self.rollout_buffer), 1)
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

    trainer = PPOTrainer(device=device)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        start_ep = trainer.load(args.resume)
        print(f"Resumed from episode {start_ep}")

    env = MockEnv()
    win_history = deque(maxlen=100)
    rollout_steps = 0
    UPDATE_EVERY = 128

    for ep in range(start_ep, args.episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = trainer.select_action(obs)
            next_obs, reward, done = env.step(action)
            trainer.store(obs, action, log_prob, value, reward, done)
            obs = next_obs
            ep_reward += reward
            rollout_steps += 1

            if rollout_steps >= UPDATE_EVERY:
                metrics = trainer.update()
                rollout_steps = 0

        win = ep_reward > 0
        win_history.append(float(win))

        if ep % 100 == 0:
            win_rate = sum(win_history) / len(win_history) if win_history else 0
            print(f"[EP {ep:5d}] reward={ep_reward:.2f} | win_rate(100)={win_rate:.2%}")

        if (ep + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_ep{ep+1}.pt")
            trainer.save(save_path, ep + 1)

    final_path = os.path.join(args.save_dir, "model_final.pt")
    trainer.save(final_path, args.episodes)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="../models")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    train(args)
