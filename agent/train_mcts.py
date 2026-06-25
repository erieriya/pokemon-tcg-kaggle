"""
agent/mcts.py(PUCTツリーMCTS)を使った自己対戦データ生成 + AlphaZero式学習ループ。

データ収集はマルチプロセスで並列化する(cg Engineのsearch APIはPythonプロセス単位の
グローバル状態(agent_ptr)を持つため、スレッドではなく別プロセスに分ける必要がある)。
各ワーカーはCPU上でネットワークの推論を行う(モデルが小さいためCPU推論で十分速く、
GPUのマルチプロセス共有による複雑さを避けられる)。学習(バックプロパゲーション)は
メインプロセスでGPUを使ってまとめて行う。

使い方:
  cd agent && uv run python train_mcts.py --games 100 --workers 16 --simulations 32 \
      --candidates 4 --epochs 3 --out models/mcts_gen1.pt [--resume models/bc_pretrain_v1.pt]
"""

import argparse
import multiprocessing as mp
import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from rl_agent import PTCGNet, encode_actions, encode_state  # noqa: E402
from train_ppo import _sequential_log_prob, read_deck  # noqa: E402
import mcts  # noqa: E402


def play_one_game(
    net_state_dict: dict,
    my_deck: list[int],
    n_simulations: int,
    num_candidates: int,
    max_steps: int,
):
    """1試合をMCTSで両陣営とも進め、((obs_dict, policy_target, player_idx)のリスト, 試合結果)を返す。
    ワーカープロセス内でCPU上にネットワークを再構築して使う。

    torch.set_num_threads(1)が重要: デフォルト(CPUコア数分)のスレッド数だと、
    バッチサイズ1の小さい推論を何度も呼ぶこのループでスレッド生成・同期のオーバーヘッドが
    支配的になり、実測で約48倍遅くなる(8回探索で2.1秒→0.044秒)。複数ワーカープロセスを
    並列実行する場合は特に深刻(各プロセスが全コア分のスレッドを要求し合って競合する)。
    """
    torch.set_num_threads(1)
    net = PTCGNet()
    net.load_state_dict(net_state_dict)
    net.eval()

    obs_dict, _ = battle_start(my_deck, my_deck)
    records = []
    steps = 0
    while steps < max_steps:
        state = obs_dict.get("current") or {}
        if state.get("result", -1) != -1:
            break
        sel = obs_dict.get("select")
        if sel is None:
            break
        options = sel.get("option") or []
        n = len(options)
        if n == 0:
            action = []
        else:
            player_idx = state.get("yourIndex", 0)
            action, policy_target = mcts.search_policy(
                obs_dict,
                net,
                my_deck,
                device="cpu",
                n_simulations=n_simulations,
                num_candidates=num_candidates,
            )
            if action is None:
                k = max(sel.get("minCount", 1) or 1, min(sel.get("maxCount", 1) or 1, n))
                action = list(range(k))
            else:
                records.append((obs_dict, policy_target, player_idx))
        try:
            obs_dict = battle_select(action)
        except IndexError:
            break
        steps += 1
    result = (obs_dict.get("current") or {}).get("result", -1)
    battle_finish()
    return records, result


def _worker(args):
    net_state_dict, my_deck, n_simulations, num_candidates, max_steps, n_games_for_worker, seed = args
    random.seed(seed)
    samples = []
    wins = losses = draws = 0
    for _ in range(n_games_for_worker):
        records, result = play_one_game(net_state_dict, my_deck, n_simulations, num_candidates, max_steps)
        if result == 0:
            wins += 1
        elif result == 1:
            losses += 1
        else:
            draws += 1
        for obs_dict, policy_target, player_idx in records:
            if result in (0, 1):
                outcome = 1.0 if result == player_idx else -1.0
            else:
                outcome = 0.0
            samples.append((obs_dict, policy_target, outcome))
    return samples, wins, losses, draws


def collect_self_play(
    net: PTCGNet,
    my_deck: list[int],
    n_games: int,
    workers: int,
    n_simulations: int,
    num_candidates: int,
    max_steps: int,
):
    net_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    workers = max(1, min(workers, n_games))
    base, extra = divmod(n_games, workers)
    games_per_worker = [base + (1 if i < extra else 0) for i in range(workers)]
    tasks = [
        (net_state_dict, my_deck, n_simulations, num_candidates, max_steps, g, i)
        for i, g in enumerate(games_per_worker)
        if g > 0
    ]
    if len(tasks) == 1:
        results = [_worker(tasks[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
            results = pool.map(_worker, tasks)

    samples = []
    total_wins = total_losses = total_draws = 0
    for worker_samples, wins, losses, draws in results:
        samples.extend(worker_samples)
        total_wins += wins
        total_losses += losses
        total_draws += draws
    print(
        f"[MCTS] collected {len(samples)} samples from {n_games} games "
        f"(player0 perspective: wins={total_wins} losses={total_losses} draws={total_draws})"
    )
    return samples


def _policy_target_to_vector(policy_target, n_options: int) -> torch.Tensor:
    vec = torch.zeros(n_options)
    for action, prob in policy_target:
        if len(action) == 1 and 0 <= action[0] < n_options:
            vec[action[0]] += prob
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def train(samples, net: PTCGNet, optimizer, epochs: int, device: str, value_coef: float, accum_steps: int):
    net.to(device)
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = total_policy_loss = total_value_loss = 0.0
        count = 0
        optimizer.zero_grad()
        accum_count = 0
        for obs_dict, policy_target, outcome in samples:
            sel = obs_dict.get("select") or {}
            options = sel.get("option") or []
            if not options or not policy_target:
                continue
            n = len(options)
            state = encode_state(obs_dict, device)
            action_feats = encode_actions(options, obs_dict, device)
            logits, value = net(state, action_feats)

            if all(len(a) == 1 for a, _ in policy_target):
                target_vec = _policy_target_to_vector(policy_target, n).to(device)
                log_probs = F.log_softmax(logits[0], dim=-1)
                policy_loss = -(target_vec * log_probs).sum()
            else:
                best_action = max(policy_target, key=lambda x: x[1])[0]
                valid_action = [i for i in best_action if 0 <= i < n]
                if not valid_action:
                    continue
                log_prob, _entropy = _sequential_log_prob(logits[0], valid_action)
                policy_loss = -log_prob

            value_loss = F.mse_loss(value.view(()), torch.tensor(outcome, device=device))
            loss = (policy_loss + value_coef * value_loss) / accum_steps

            loss.backward()
            accum_count += 1

            total_loss += loss.item() * accum_steps
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            count += 1
            if accum_count >= accum_steps:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        denom = max(1, count)
        print(
            f"[MCTS-train] epoch={epoch} avg_loss={total_loss/denom:.4f} "
            f"avg_policy_loss={total_policy_loss/denom:.4f} avg_value_loss={total_value_loss/denom:.4f} "
            f"samples_used={count}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=32, help="1意思決定あたりのMCTSシミュレーション数")
    parser.add_argument("--candidates", type=int, default=4, help="ルートで評価する候補手の数")
    parser.add_argument("--max_steps", type=int, default=300, help="1試合あたりの最大ステップ数")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--accum_steps", type=int, default=32)
    parser.add_argument("--resume", type=str, default="", help="初期重み(BCやPPOのチェックポイント)")
    parser.add_argument("--out", type=str, default="models/mcts_gen1.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    my_deck = read_deck("deck.csv")

    net = PTCGNet()
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ckpt["model"])
        print(f"[MCTS] resumed weights from {args.resume}")

    print(f"[MCTS] collecting {args.games} self-play games with {args.workers} workers...")
    samples = collect_self_play(
        net, my_deck, args.games, args.workers, args.simulations, args.candidates, args.max_steps
    )

    net.to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train(samples, net, optimizer, args.epochs, args.device, args.value_coef, args.accum_steps)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "episode": 0},
        args.out,
    )
    print(f"[MCTS] saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
