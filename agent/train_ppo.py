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
from collections import Counter, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_start, battle_select, battle_finish

from rl_agent import PTCGNet, encode_state, encode_actions
import lucario_v1_agent
import lucario_v2_agent
import crustle_agent
import iono_agent
import abomasnow_agent
import alakazam_agent
import archaludon_agent

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sanitize_opponent_action(action, sel: dict, n: int) -> list[int]:
    """固定対戦相手(opponent_fn)が返したactionを検証し、不正ならエンジンが受理する
    合法手にフォールバックする。

    型・範囲チェックだけでは不十分（lucario_v1_agentがminCount=0の場面でも1個選んで
    しまうバグで実際にIndexErrorを起こした）。重複の有無とminCount/maxCountの個数も
    検証する。
    """
    min_c = sel.get("minCount", 0) or 0
    max_c = sel.get("maxCount", 0) or 0
    valid = (
        isinstance(action, list)
        and all(isinstance(a, int) and 0 <= a < n for a in action)
        and len(set(action)) == len(action)
        and min_c <= len(action) <= max_c
    )
    if valid:
        return action
    k = max(min_c, min(max_c, n))
    return list(range(k))


CRASH_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "crash_diagnostics.jsonl")


def _dump_crash_diagnostics(tag: str, obs_dict: dict, action, sel: dict) -> None:
    """battle_selectがIndexErrorで拒否した時の状況を logs/crash_diagnostics.jsonl に追記する。

    minCount/maxCountが0の場面を1に書き換えるバグ等、過去に複数回このIndexErrorで
    学習runがクラッシュしたため、次に未知の原因で起きた時に原因をすぐ特定できるよう
    詳細を残す(エピソード自体は1つ捨てて学習は継続させる)。
    """
    try:
        import json as _json
        import time as _time

        record = {
            "time": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag,
            "context": sel.get("context") if sel else None,
            "minCount": sel.get("minCount") if sel else None,
            "maxCount": sel.get("maxCount") if sel else None,
            "n_options": len(sel.get("option") or []) if sel else None,
            "option_types": [o.get("type") for o in (sel.get("option") or [])] if sel else None,
            "action": action,
        }
        os.makedirs(os.path.dirname(CRASH_LOG_PATH), exist_ok=True)
        with open(CRASH_LOG_PATH, "a") as f:
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[WARN] crash diagnostics dump failed: {exc}")


def random_opponent(obs_dict: dict) -> list[int]:
    """完全ランダムな合法手エージェント（固定対戦相手プールの最弱メンバー）。"""
    sel = obs_dict.get("select")
    if sel is None:
        return []
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return []
    max_c = sel.get("maxCount", 0) or 0
    min_c = sel.get("minCount", 0) or 0
    k = max(min_c, min(max_c, n))
    return random.sample(range(n), k) if k else []


# 固定対戦相手プール: 名前 -> (agent関数 obs_dict->list[int], デッキファイル名)
# "selfplay" は特別扱い(学習中ネットの自己対戦)で、ここには含めない。
FIXED_OPPONENTS = {
    "random": (random_opponent, None),
    "lucario_v1": (lucario_v1_agent.agent, "deck_lucario_v1.csv"),
    "lucario_v2": (lucario_v2_agent.agent, "deck_lucario_v2.csv"),
    "crustle": (crustle_agent.agent, "deck_crustle.csv"),
    "iono": (iono_agent.agent, "deck_iono.csv"),
    "abomasnow": (abomasnow_agent.agent, "deck_abomasnow.csv"),
    "alakazam": (alakazam_agent.agent, "deck_alakazam.csv"),
    "archaludon": (archaludon_agent.agent, "deck_archaludon.csv"),
}


def _sequential_sample(logits: torch.Tensor, k: int) -> tuple[list[int], torch.Tensor]:
    """重複なしでk個選ぶ（選んだ選択肢を毎回-infでマスクして再softmax）。
    minCount/maxCountが1より大きいselect（ベンチに複数出す等）に対応するため。

    以前は`logits[idx] = -1e9`という大きな負の定数で代用していたが、学習が進んで
    元のlogitsの絶対値が大きくなる場面では「選んだ方が必ずしも一番小さい値になる」
    保証がなく、実際に重複したindexを選んでしまいエンジンにIndexErrorで拒否される
    クラッシュが本番run中に発生した。booleanマスク+`-inf`で確率を完全に0にすることで
    logitsの大きさに関わらず再選択を構造的に禁止する。
    """
    logits = logits.clone()
    n = logits.shape[-1]
    available = torch.ones(n, dtype=torch.bool, device=logits.device)
    chosen: list[int] = []
    total_log_prob = torch.zeros((), device=logits.device)
    for _ in range(k):
        masked_logits = logits.masked_fill(~available, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        total_log_prob = total_log_prob + dist.log_prob(idx)
        chosen.append(idx.item())
        available[idx] = False
    return chosen, total_log_prob


def _sequential_log_prob(logits: torch.Tensor, action_idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """指定した順序でindexを選んだ場合の対数確率合計とエントロピー合計（PPO更新時に使用）。

    _sequential_sampleと同じ理由でbooleanマスク+`-inf`を使う（ロールアウト時と更新時で
    同じマスク方式に揃えておかないと、極端なlogitsの場面でlog_probの計算が食い違う）。
    """
    logits = logits.clone()
    n = logits.shape[-1]
    available = torch.ones(n, dtype=torch.bool, device=logits.device)
    total_log_prob = torch.zeros((), device=logits.device)
    total_entropy = torch.zeros((), device=logits.device)
    for idx in action_idxs:
        masked_logits = logits.masked_fill(~available, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        total_log_prob = total_log_prob + dist.log_prob(torch.tensor(idx, device=logits.device))
        total_entropy = total_entropy + dist.entropy()
        available[idx] = False
    return total_log_prob, total_entropy


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


LOG_TYPE_RESULT = 23
# State.RESULT.reason: 1=サイド0枚 2=デッキアウト 3=場のポケモン0 4=カード効果
REASON_DECKOUT = 2
REASON_WIPEOUT = 3


def _result_reason(obs_dict: dict) -> int | None:
    """試合終了時のobs_dictからRESULTログのreasonを取り出す（デッキアウト勝ちかどうかの判定用）。"""
    for log in obs_dict.get("logs") or []:
        if log.get("type") == LOG_TYPE_RESULT:
            return log.get("reason")
    return None


def _prize_count(obs_dict: dict, player_idx: int) -> int | None:
    """player_idxの残りサイド枚数（取るたびに減る）。取得できなければNone。"""
    players = (obs_dict.get("current") or {}).get("players") or []
    if player_idx >= len(players):
        return None
    return len(players[player_idx].get("prize") or [])


# ============================================================
# Self-play環境（cgエンジンを直接ラップ）
# ============================================================

class PTCGSelfPlayEnv:
    """同じpolicyが両プレイヤーを操作する自己対戦環境。

    deck.csv固定デッキでのミラーマッチ（research.md Phase3でデッキ可変化はTODO）。
    """

    def __init__(
        self,
        deck_path: str,
        prize_rewards: list[float],
        deckout_win_reward: float = 0.5,
        deckout_loss_reward: float = -1.0,
        wipeout_loss_reward: float = -1.0,
    ):
        self.deck = read_deck(deck_path)
        self.prize_rewards = prize_rewards  # 1枚目, 2枚目, ... 取得順ごとの中間報酬
        self.deckout_win_reward = deckout_win_reward
        self.deckout_loss_reward = deckout_loss_reward  # 自分がデッキアウトして負けた時の罰則
        self.wipeout_loss_reward = wipeout_loss_reward  # 自分の場のポケモンが0になって負けた時の罰則

    def play_episode(self, trainer: "PPOTrainer", max_steps: int = 2000) -> dict:
        """1試合実行し、両プレイヤー視点の遷移をtrainerのバッファへ積む。

        各プレイヤーは自分の手番の連続をそれぞれ独立した軌跡として記録し、
        試合終了時にそのプレイヤー自身の最後の手番にだけ勝敗報酬を入れる
        （手番が交互に来るため、相手の行動を挟んでも各プレイヤーのGAEは
        そのプレイヤー自身の軌跡内だけで計算される）。

        サイドを取った瞬間に勝敗報酬より弱い中間報酬を加算する（何枚目に取ったかでprize_rewards[i]
        を参照するため、終盤のサイドを大きくする設定にすればフィニッシュへの圧力を強められる）。
        デッキアウト勝ちは通常の勝利（KOでサイドを取り切る/相手の場のポケモンを0にする）
        より報酬を弱める（先攻はドローをスキップする分デッキアウトで勝ちやすく、何もせず
        待つだけの退化戦略が強化されてしまうのを防ぐ）。逆に自分がデッキアウト/全滅して
        負けた場合は通常のKO負けより罰則を強め、消極的なプレイで負けることを避けさせる。
        """
        obs_dict, _ = battle_start(self.deck, self.deck)
        last_idx = {0: None, 1: None}
        prize_count = {0: _prize_count(obs_dict, 0), 1: _prize_count(obs_dict, 1)}
        total_prizes = dict(prize_count)  # 試合開始時の残りサイド枚数（=何枚目を取ったかの基準）
        step = 0
        result = -1
        start_len = {0: len(trainer.buffers[0]), 1: len(trainer.buffers[1])}

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

            try:
                obs_dict = battle_select(action)
            except IndexError:
                # エンジンがactionを不正と判断して拒否した(過去に複数回原因の異なるバグで
                # 発生済み)。このエピソードだけ捨てて学習runそのものは継続させる。
                _dump_crash_diagnostics("play_episode", obs_dict, action, sel)
                for p in (0, 1):
                    del trainer.buffers[p][start_len[p]:]
                try:
                    battle_finish()
                except Exception:
                    pass
                return {"result": -1, "steps": step, "turns": state.get("turn", 0), "reason": None, "aborted": True}
            step += 1

            new_count = _prize_count(obs_dict, player_idx)
            prev_count = prize_count[player_idx]
            if new_count is not None and prev_count is not None and new_count < prev_count:
                taken = prev_count - new_count
                already_taken = (total_prizes[player_idx] or 0) - prev_count
                reward_gain = sum(
                    self.prize_rewards[i]
                    for i in range(already_taken, already_taken + taken)
                    if 0 <= i < len(self.prize_rewards)
                )
                buf_idx = last_idx[player_idx]
                trainer.buffers[player_idx][buf_idx]["reward"] += reward_gain
            for p in (0, 1):
                pc = _prize_count(obs_dict, p)
                if pc is not None:
                    prize_count[p] = pc

        result = (obs_dict.get("current") or {}).get("result", -1)
        turn = (obs_dict.get("current") or {}).get("turn", 0)
        reason = _result_reason(obs_dict)
        battle_finish()

        for player_idx, idx in last_idx.items():
            if idx is None:
                continue
            if result == player_idx:
                reward = self.deckout_win_reward if reason == REASON_DECKOUT else 1.0
            elif result in (0, 1):
                if reason == REASON_DECKOUT:
                    reward = self.deckout_loss_reward
                elif reason == REASON_WIPEOUT:
                    reward = self.wipeout_loss_reward
                else:
                    reward = -1.0
            else:
                reward = 0.0
            buf = trainer.buffers[player_idx][idx]
            buf["reward"] += reward
            buf["done"] = True

        return {"result": result, "steps": step, "turns": turn, "reason": reason}

    def play_episode_vs_opponent(
        self,
        trainer: "PPOTrainer",
        opponent_fn,
        opponent_deck: list[int],
        max_steps: int = 2000,
    ) -> dict:
        """学習中ネット(片側のみ)を固定の対戦相手(opponent_fn)と対戦させる。

        play_episode（自己対戦、両プレイヤーがtrainerのバッファに記録される）と異なり、
        ここではtrainer側の手番だけがバッファに記録され、opponent_fn側の手番は勾度なしで
        即時に行動を返すだけ。報酬ロジック（サイド中間報酬・デッキアウト/全滅の罰則）は
        play_episodeと同一にし、対戦相手の種類による報酬設計の差が出ないようにする。
        先攻/後攻の構造的アドバンテージを均すため、毎試合trainer側の座席をランダム化する。
        """
        my_idx = random.randint(0, 1)
        deck0 = self.deck if my_idx == 0 else opponent_deck
        deck1 = opponent_deck if my_idx == 0 else self.deck
        obs_dict, _ = battle_start(deck0, deck1)

        last_idx = None
        my_prize = _prize_count(obs_dict, my_idx)
        total_prizes = my_prize
        step = 0
        result = -1
        start_len = len(trainer.buffers[my_idx])

        while step < max_steps:
            state = obs_dict.get("current") or {}
            result = state.get("result", -1)
            sel = obs_dict.get("select")
            if result != -1 or sel is None:
                break

            player_idx = state.get("yourIndex", 0)
            if player_idx == my_idx:
                action, log_prob, value = trainer.select_action(obs_dict)
                trainer.store(my_idx, obs_dict, action, log_prob, value, reward=0.0, done=False)
                last_idx = len(trainer.buffers[my_idx]) - 1
            else:
                options = sel.get("option") or []
                n = len(options)
                action = opponent_fn(obs_dict) if n else []
                action = _sanitize_opponent_action(action, sel, n)

            try:
                obs_dict = battle_select(action)
            except IndexError:
                # play_episodeと同じ安全策(このエピソードだけ捨てて学習runは継続)。
                _dump_crash_diagnostics("play_episode_vs_opponent", obs_dict, action, sel)
                del trainer.buffers[my_idx][start_len:]
                try:
                    battle_finish()
                except Exception:
                    pass
                return {
                    "result": -1, "steps": step, "turns": state.get("turn", 0),
                    "reason": None, "aborted": True, "my_idx": my_idx,
                }
            step += 1

            new_count = _prize_count(obs_dict, my_idx)
            if new_count is not None and my_prize is not None and new_count < my_prize and last_idx is not None:
                taken = my_prize - new_count
                already_taken = (total_prizes or 0) - my_prize
                reward_gain = sum(
                    self.prize_rewards[i]
                    for i in range(already_taken, already_taken + taken)
                    if 0 <= i < len(self.prize_rewards)
                )
                trainer.buffers[my_idx][last_idx]["reward"] += reward_gain
            if new_count is not None:
                my_prize = new_count

        result = (obs_dict.get("current") or {}).get("result", -1)
        turn = (obs_dict.get("current") or {}).get("turn", 0)
        reason = _result_reason(obs_dict)
        battle_finish()

        if last_idx is not None:
            if result == my_idx:
                reward = self.deckout_win_reward if reason == REASON_DECKOUT else 1.0
            elif result in (0, 1):
                if reason == REASON_DECKOUT:
                    reward = self.deckout_loss_reward
                elif reason == REASON_WIPEOUT:
                    reward = self.wipeout_loss_reward
                else:
                    reward = -1.0
            else:
                reward = 0.0
            buf = trainer.buffers[my_idx][last_idx]
            buf["reward"] += reward
            buf["done"] = True

        return {"result": result, "steps": step, "turns": turn, "reason": reason, "my_idx": my_idx}


def play_eval_game(
    net: "PTCGNet",
    my_deck: list[int],
    opponent_fn,
    opponent_deck: list[int],
    device: str,
    max_steps: int = 2000,
) -> tuple[int, int]:
    """学習中のpolicy(greedy, no_grad)とopponent_fnを1戦対戦させ(結果, netのplayer_idx)を返す。

    cgエンジンは常にbattle_startの第1引数側player_idx=0が先攻になる（コイントスではない）ため、
    net側を毎回player_idx 0に固定すると評価が常に先攻有利になり、デッキアウト勝ちで
    勝率が見かけ上高くなる。net側のplayer_idxをランダム化して先攻後攻を均等にする。
    """
    net_idx = random.randint(0, 1)
    deck0 = my_deck if net_idx == 0 else opponent_deck
    deck1 = opponent_deck if net_idx == 0 else my_deck
    obs_dict, _ = battle_start(deck0, deck1)
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

        if player_idx == net_idx:
            # "or 1" は使わない: maxCount/minCountが正当に0の場面(強制選択なし)を
            # 1に書き換えてしまい、エンジン側がIndexErrorで拒否する原因になる。
            max_c = sel.get("maxCount", 1)
            min_c = sel.get("minCount", 1)
            k = max(min_c, min(max_c, n))
            if n == 0:
                action = []
            else:
                with torch.no_grad():
                    s = encode_state(obs_dict, device)
                    a = encode_actions(options, obs_dict, device)
                    logits, _ = net(s, a)
                    probs = F.softmax(logits[0], dim=-1)
                action = torch.topk(probs, k).indices.tolist()
        else:
            action = opponent_fn(obs_dict) if n else []
            action = _sanitize_opponent_action(action, sel, n)

        obs_dict = battle_select(action)
        step += 1

    result = (obs_dict.get("current") or {}).get("result", -1)
    battle_finish()
    return result, net_idx


def evaluate(
    net: "PTCGNet",
    my_deck: list[int],
    fixed_opponents: list[tuple[str, object, list[int]]],
    device: str,
    n_games: int = 10,
) -> dict[str, float]:
    """固定対戦相手それぞれについて、学習中policy(greedy)の勝率を返す。

    fixed_opponentsは(名前, agent関数, デッキ)のリスト（"selfplay"は含めない。
    両陣営とも同じ重みのため、mirrorの勝率は単なるドロー変動を測るだけで
    学習が進んでいるかの指標にならず、自己対戦の質は学習ループ側のローリング
    統計(avg_turns/deckout_win_rate)で別途見ているため）。
    """
    results: dict[str, float] = {}
    for name, opponent_fn, opponent_deck in fixed_opponents:
        wins = 0
        played = 0
        for _ in range(n_games):
            result, net_idx = play_eval_game(net, my_deck, opponent_fn, opponent_deck, device)
            if result not in (0, 1):
                continue
            played += 1
            if result == net_idx:
                wins += 1
        results[name] = wins / played if played else 0.0
    return results


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

        # "or 1" は使わない: maxCount/minCountが正当に0の場面(強制選択なし)を
        # 1に書き換えてしまい、エンジン側がIndexErrorで拒否する原因になる
        # (20000エピソードrunがEP5340付近でこれにより実際にクラッシュした)。
        max_count = select.get("maxCount", 1)
        min_count = select.get("minCount", 1)
        k = max(min_count, min(max_count, n))

        state = encode_state(obs_dict, self.device)
        action_feats = encode_actions(options, obs_dict, self.device)

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
                action_feats = encode_actions(options, buf["obs"], self.device)
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
        self.net.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                print(f"[LOAD] optimizer state skipped: {exc}")
        return ckpt.get("episode", 0)


# ============================================================
# 学習ループ
# ============================================================

def parse_prize_rewards(s: str) -> list[float]:
    """\"0.05,0.05,0.08,0.1,0.15,0.25\" のようなカンマ区切り文字列を、
    1枚目〜N枚目に取ったサイドごとの中間報酬リストに変換する。"""
    rewards = [float(x) for x in s.split(",") if x.strip() != ""]
    if not rewards:
        raise ValueError(f"--prize_rewards が空です: {s!r}")
    return rewards


def build_opponent_pool(args, default_deck: list[int]) -> list[tuple[str, object, list[int] | None, float]]:
    """(名前, agent関数 or None, デッキ or None, サンプリング重み) のリストを作る。

    "selfplay" は agent関数・デッキともにNone（PTCGSelfPlayEnv.play_episodeを直接使う特別扱い）。
    randomエージェントは専用デッキを持たないため、自己対戦と同じdefault_deck（agent/deck.csv）を使う。
    重み0の相手はプールから除外する。
    """
    weights = {
        "selfplay": args.selfplay_weight,
        "random": args.random_weight,
        "lucario_v1": args.lucario_v1_weight,
        "lucario_v2": args.lucario_v2_weight,
        "crustle": args.crustle_weight,
        "iono": args.iono_weight,
        "abomasnow": args.abomasnow_weight,
    }
    pool: list[tuple[str, object, list[int] | None, float]] = []
    if weights["selfplay"] > 0:
        pool.append(("selfplay", None, None, weights["selfplay"]))
    for name, (agent_fn, deck_filename) in FIXED_OPPONENTS.items():
        w = weights[name]
        if w <= 0:
            continue
        deck = read_deck(os.path.join(AGENT_DIR, deck_filename)) if deck_filename else default_deck
        pool.append((name, agent_fn, deck, w))
    if not pool:
        raise ValueError("対戦相手プールが空です。weight引数のいずれかを正の値にしてください。")
    return pool


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    prize_rewards = parse_prize_rewards(args.prize_rewards)
    print(
        f"prize_rewards={prize_rewards} deckout_win_reward={args.deckout_win_reward} "
        f"deckout_loss_reward={args.deckout_loss_reward} wipeout_loss_reward={args.wipeout_loss_reward}"
    )

    deck_path = os.path.join(os.path.dirname(__file__), "deck.csv")
    trainer = PPOTrainer(device=device)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        start_ep = trainer.load(args.resume)
        print(f"Resumed from episode {start_ep}")

    env = PTCGSelfPlayEnv(
        deck_path,
        prize_rewards=prize_rewards,
        deckout_win_reward=args.deckout_win_reward,
        deckout_loss_reward=args.deckout_loss_reward,
        wipeout_loss_reward=args.wipeout_loss_reward,
    )

    opponent_pool = build_opponent_pool(args, default_deck=env.deck)
    pool_names = [p[0] for p in opponent_pool]
    pool_weights = [p[3] for p in opponent_pool]
    print("opponent_pool: " + ", ".join(f"{n}={w:g}" for n, w in zip(pool_names, pool_weights)))
    # evaluate()用: 学習で実際に使っている固定対戦相手だけを対象にする("selfplay"は除く)。
    fixed_eval_opponents = [(n, fn, deck) for n, fn, deck, _ in opponent_pool if n != "selfplay"]
    turn_history = deque(maxlen=100)
    draw_history = deque(maxlen=100)
    deckout_history = deque(maxlen=100)
    # 固定対戦相手ごとの直近勝率（自己対戦は同一ネット同士のため対象外）
    opponent_win_history = {name: deque(maxlen=100) for name in pool_names if name != "selfplay"}
    opponent_pick_history = deque(maxlen=200)  # 直近のサンプリング内訳確認用
    UPDATE_EVERY = args.update_every
    MAX_CONSECUTIVE_ABORTS = 20  # battle_selectのIndexErrorが連発した場合に学習runを止める閾値
    consecutive_aborts = 0

    for ep in range(start_ep, args.episodes):
        name, opp_fn, opp_deck, _ = random.choices(opponent_pool, weights=pool_weights, k=1)[0]
        opponent_pick_history.append(name)
        if name == "selfplay":
            info = env.play_episode(trainer, max_steps=args.max_steps)
        else:
            info = env.play_episode_vs_opponent(trainer, opp_fn, opp_deck, max_steps=args.max_steps)

        if info.get("aborted"):
            # エンジンにactionを拒否されたエピソード（バッファは既にplay_episode側で巻き戻し済み）。
            # 統計には数えず1試合分捨てて次へ進む。連発する場合は未知のバグの可能性が高いので止める。
            consecutive_aborts += 1
            print(f"[EP {ep:5d}] aborted ({name}): battle_select rejected an action, see logs/crash_diagnostics.jsonl")
            if consecutive_aborts >= MAX_CONSECUTIVE_ABORTS:
                raise RuntimeError(
                    f"{consecutive_aborts}エピソード連続でbattle_selectがIndexErrorを起こしました。"
                    f"logs/crash_diagnostics.jsonlを確認してください。"
                )
            continue
        consecutive_aborts = 0

        if name != "selfplay":
            opponent_win_history[name].append(1.0 if info["result"] == info["my_idx"] else 0.0)

        turn_history.append(info["turns"])
        draw_history.append(1.0 if info["result"] not in (0, 1) else 0.0)
        deckout_history.append(1.0 if info["result"] in (0, 1) and info["reason"] == REASON_DECKOUT else 0.0)

        if trainer.n_stored() >= UPDATE_EVERY:
            trainer.update()

        if ep % args.log_interval == 0:
            avg_turns = sum(turn_history) / len(turn_history) if turn_history else 0
            draw_rate = sum(draw_history) / len(draw_history) if draw_history else 0
            deckout_rate = sum(deckout_history) / len(deckout_history) if deckout_history else 0
            pick_counts = Counter(opponent_pick_history)
            pick_str = " ".join(f"{n}={pick_counts.get(n, 0)}" for n in pool_names)
            win_str = " ".join(
                f"{n}_win={sum(h) / len(h):.1%}" if h else f"{n}_win=-"
                for n, h in opponent_win_history.items()
            )
            print(
                f"[EP {ep:5d}] avg_turns(100)={avg_turns:.1f} draw_rate(100)={draw_rate:.1%} "
                f"deckout_win_rate(100)={deckout_rate:.1%} | picks(200): {pick_str} | {win_str}"
            )

        if args.eval_interval and ep > 0 and ep % args.eval_interval == 0:
            eval_results = evaluate(trainer.net, env.deck, fixed_eval_opponents, device, n_games=args.eval_games)
            eval_str = " ".join(f"{n}={wr:.1%}" for n, wr in eval_results.items())
            print(f"[EP {ep:5d}] eval winrate({args.eval_games} games each): {eval_str}")

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
    parser.add_argument(
        "--prize_rewards", type=str, default="0.05,0.05,0.08,0.1,0.15,0.25",
        help="サイドを1枚取った時の中間報酬を取得順にカンマ区切りで指定（勝敗の±1.0より弱める）。"
             "デフォルトは終盤のサイドほど大きくしてフィニッシュを後押しする設定。",
    )
    parser.add_argument("--deckout_win_reward", type=float, default=0.5, help="デッキアウト勝ちの報酬（KO等の通常勝利は1.0固定）")
    parser.add_argument("--deckout_loss_reward", type=float, default=-1.5, help="自分がデッキアウトして負けた時の罰則（通常のKO負けは-1.0固定）")
    parser.add_argument("--wipeout_loss_reward", type=float, default=-1.5, help="自分の場のポケモンが0になって負けた時の罰則（通常のKO負けは-1.0固定）")
    parser.add_argument("--selfplay_weight", type=float, default=0.6, help="対戦相手プールでの自己対戦のサンプリング重み")
    parser.add_argument("--random_weight", type=float, default=0.05, help="対戦相手プールでのランダムエージェントのサンプリング重み")
    parser.add_argument("--lucario_v1_weight", type=float, default=0.05, help="対戦相手プールでのlucario_v1(Kaggle notebook移植)のサンプリング重み")
    parser.add_argument("--lucario_v2_weight", type=float, default=0.05, help="対戦相手プールでのlucario_v2(Kaggle notebook移植)のサンプリング重み")
    parser.add_argument(
        "--crustle_weight", type=float, default=0.15,
        help="対戦相手プールでのcrustle(Crustleウォール, Kaggle notebook移植)のサンプリング重み。"
             "ex/megaExアタッカーを無効化する初日メタの最重要対策のため他の固定相手より高めのデフォルト。",
    )
    parser.add_argument("--iono_weight", type=float, default=0.05, help="対戦相手プールでのiono(Bellibolt ex, kiyotah公式サンプル移植)のサンプリング重み")
    parser.add_argument("--abomasnow_weight", type=float, default=0.05, help="対戦相手プールでのabomasnow(Mega Abomasnow ex, kiyotah公式サンプル移植)のサンプリング重み")
    args = parser.parse_args()
    train(args)
