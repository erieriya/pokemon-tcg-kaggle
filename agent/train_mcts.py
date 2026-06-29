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

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
CG_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_submission")
if os.path.exists(CG_PATH):
    sys.path.insert(0, CG_PATH)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from rl_agent import PTCGNet, encode_actions, encode_state  # noqa: E402
from train_ppo import _sequential_log_prob, read_deck, FIXED_OPPONENTS  # noqa: E402
import mcts  # noqa: E402


def _sample_action_by_visit(policy_target, temperature: float):
    if not policy_target:
        return None
    if temperature <= 0:
        return max(policy_target, key=lambda x: x[1])[0]

    weights = [max(prob, 0.0) ** (1.0 / temperature) for _, prob in policy_target]
    total = sum(weights)
    if total <= 0:
        return max(policy_target, key=lambda x: x[1])[0]
    return random.choices([action for action, _ in policy_target], weights=weights, k=1)[0]


def _state_dict_to_numpy(state_dict: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _state_dict_from_numpy(state_dict: dict) -> dict:
    return {k: torch.from_numpy(v) for k, v in state_dict.items()}


def play_one_game(
    net_state_dict: dict,
    my_deck: list[int],
    n_simulations: int,
    num_candidates: int,
    min_candidates: int,
    dynamic_candidates: bool,
    max_steps: int,
    temperature: float = 1.0,
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
    if net_state_dict and not isinstance(next(iter(net_state_dict.values())), torch.Tensor):
        net_state_dict = _state_dict_from_numpy(net_state_dict)
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
                min_candidates=min_candidates,
                dynamic_candidates=dynamic_candidates,
            )
            if action is None:
                k = max(sel.get("minCount", 1) or 1, min(sel.get("maxCount", 1) or 1, n))
                action = list(range(k))
            else:
                action = _sample_action_by_visit(policy_target, temperature)
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
    (
        net_state_dict,
        my_deck,
        n_simulations,
        num_candidates,
        min_candidates,
        dynamic_candidates,
        max_steps,
        temperature,
        n_games_for_worker,
        seed,
    ) = args
    random.seed(seed)
    samples = []
    wins = losses = draws = 0
    for _ in range(n_games_for_worker):
        records, result = play_one_game(
            net_state_dict,
            my_deck,
            n_simulations,
            num_candidates,
            min_candidates,
            dynamic_candidates,
            max_steps,
            temperature,
        )
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
    min_candidates: int,
    dynamic_candidates: bool,
    max_steps: int,
    temperature: float,
):
    net_state_dict = _state_dict_to_numpy(net.state_dict())
    workers = max(1, min(workers, n_games))
    base, extra = divmod(n_games, workers)
    games_per_worker = [base + (1 if i < extra else 0) for i in range(workers)]
    tasks = [
        (
            net_state_dict,
            my_deck,
            n_simulations,
            num_candidates,
            min_candidates,
            dynamic_candidates,
            max_steps,
            temperature,
            g,
            i,
        )
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


def play_vs_opponent_game(
    net_state_dict: dict,
    my_deck: list[int],
    opponent_fn,
    opponent_deck: list[int],
    n_simulations: int,
    num_candidates: int,
    min_candidates: int,
    dynamic_candidates: bool,
    max_steps: int,
    temperature: float = 1.0,
):
    """1試合をnet(MCTS)対固定ヒューリスティック(opponent_fn)で進め、
    ((obs_dict, policy_target, net_idx)のリスト, 試合結果, net_idx)を返す。
    net側の手番のみMCTSで探索し記録する。相手側はopponent_fnを直接呼ぶだけ。
    """
    torch.set_num_threads(1)
    net = PTCGNet()
    if net_state_dict and not isinstance(next(iter(net_state_dict.values())), torch.Tensor):
        net_state_dict = _state_dict_from_numpy(net_state_dict)
    net.load_state_dict(net_state_dict)
    net.eval()

    net_idx = random.randint(0, 1)
    deck0 = my_deck if net_idx == 0 else opponent_deck
    deck1 = opponent_deck if net_idx == 0 else my_deck
    obs_dict, _ = battle_start(deck0, deck1)
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
        player_idx = state.get("yourIndex", 0)
        if n == 0:
            action = []
        elif player_idx == net_idx:
            action, policy_target = mcts.search_policy(
                obs_dict,
                net,
                my_deck,
                device="cpu",
                n_simulations=n_simulations,
                num_candidates=num_candidates,
                min_candidates=min_candidates,
                dynamic_candidates=dynamic_candidates,
            )
            if action is None:
                k = max(sel.get("minCount", 1) or 1, min(sel.get("maxCount", 1) or 1, n))
                action = list(range(k))
            else:
                action = _sample_action_by_visit(policy_target, temperature)
                records.append((obs_dict, policy_target, net_idx))
        else:
            action = opponent_fn(obs_dict)
        try:
            obs_dict = battle_select(action)
        except IndexError:
            break
        steps += 1
    result = (obs_dict.get("current") or {}).get("result", -1)
    battle_finish()
    return records, result, net_idx


def _vs_worker(args):
    (
        net_state_dict,
        my_deck,
        opponent_fn,
        opponent_deck,
        n_simulations,
        num_candidates,
        min_candidates,
        dynamic_candidates,
        max_steps,
        temperature,
        n_games_for_worker,
        seed,
    ) = args
    random.seed(seed)
    samples = []
    wins = losses = draws = 0
    for _ in range(n_games_for_worker):
        records, result, net_idx = play_vs_opponent_game(
            net_state_dict,
            my_deck,
            opponent_fn,
            opponent_deck,
            n_simulations,
            num_candidates,
            min_candidates,
            dynamic_candidates,
            max_steps,
            temperature,
        )
        if result == net_idx:
            wins += 1
        elif result in (0, 1):
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


def collect_vs_opponent(
    net: PTCGNet,
    my_deck: list[int],
    opponent_name: str,
    opponent_fn,
    opponent_deck: list[int],
    n_games: int,
    workers: int,
    n_simulations: int,
    num_candidates: int,
    min_candidates: int,
    dynamic_candidates: bool,
    max_steps: int,
    temperature: float,
):
    net_state_dict = _state_dict_to_numpy(net.state_dict())
    workers = max(1, min(workers, n_games))
    base, extra = divmod(n_games, workers)
    games_per_worker = [base + (1 if i < extra else 0) for i in range(workers)]
    tasks = [
        (
            net_state_dict,
            my_deck,
            opponent_fn,
            opponent_deck,
            n_simulations,
            num_candidates,
            min_candidates,
            dynamic_candidates,
            max_steps,
            temperature,
            g,
            i,
        )
        for i, g in enumerate(games_per_worker)
        if g > 0
    ]
    if len(tasks) == 1:
        results = [_vs_worker(tasks[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
            results = pool.map(_vs_worker, tasks)

    samples = []
    total_wins = total_losses = total_draws = 0
    for worker_samples, wins, losses, draws in results:
        samples.extend(worker_samples)
        total_wins += wins
        total_losses += losses
        total_draws += draws
    print(
        f"[MCTS] vs {opponent_name}: collected {len(samples)} samples from {n_games} games "
        f"(net perspective: wins={total_wins} losses={total_losses} draws={total_draws})"
    )
    return samples


def collect_vs_opponents(
    net: PTCGNet,
    my_deck: list[int],
    opponent_specs: list[tuple[str, object, list[int]]],
    n_games_per_opponent: int,
    workers: int,
    n_simulations: int,
    num_candidates: int,
    min_candidates: int,
    dynamic_candidates: bool,
    max_steps: int,
    temperature: float,
):
    samples = []
    for name, opponent_fn, opponent_deck in opponent_specs:
        samples += collect_vs_opponent(
            net,
            my_deck,
            name,
            opponent_fn,
            opponent_deck,
            n_games_per_opponent,
            workers,
            n_simulations,
            num_candidates,
            min_candidates,
            dynamic_candidates,
            max_steps,
            temperature,
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


def train(
    samples,
    net: PTCGNet,
    optimizer,
    epochs: int,
    device: str,
    value_coef: float,
    batch_size: int,
):
    net.to(device)
    batch_size = max(1, batch_size)
    encoded_samples = []
    for obs_dict, policy_target, outcome in samples:
        sel = obs_dict.get("select") or {}
        options = sel.get("option") or []
        if not options or not policy_target:
            continue
        state = encode_state(obs_dict, "cpu")
        action_feats = encode_actions(options, obs_dict, "cpu")
        encoded_samples.append(
            (state, action_feats, action_feats.shape[1], policy_target, outcome)
        )

    for epoch in range(epochs):
        random.shuffle(encoded_samples)
        total_loss = total_policy_loss = total_value_loss = 0.0
        count = 0
        for start in range(0, len(encoded_samples), batch_size):
            batch = encoded_samples[start:start + batch_size]
            states = []
            action_feats_list = []
            n_actions = []
            outcomes = []
            policy_targets = []
            for state, action_feats, n, policy_target, outcome in batch:
                states.append(state)
                action_feats_list.append(action_feats)
                n_actions.append(n)
                outcomes.append(outcome)
                policy_targets.append(policy_target)

            max_n = max(n_actions)
            padded_action_feats = []
            action_masks = []
            for action_feats, n in zip(action_feats_list, n_actions):
                if n < max_n:
                    pad = torch.zeros(
                        1,
                        max_n - n,
                        action_feats.shape[-1],
                        dtype=action_feats.dtype,
                        device=action_feats.device,
                    )
                    action_feats = torch.cat([action_feats, pad], dim=1)
                padded_action_feats.append(action_feats)
                mask = torch.zeros(1, max_n, dtype=torch.bool)
                mask[:, :n] = True
                action_masks.append(mask)

            state_batch = {
                key: torch.cat([state[key] for state in states], dim=0).to(device)
                for key in states[0]
            }
            action_feats_batch = torch.cat(padded_action_feats, dim=0).to(device)
            action_mask_batch = torch.cat(action_masks, dim=0).to(device)
            logits, value = net(state_batch, action_feats_batch, action_mask_batch)

            policy_losses = []
            used_indices = []
            for i, policy_target in enumerate(policy_targets):
                n = n_actions[i]
                sample_logits = logits[i, :n]
                if all(len(a) == 1 for a, _ in policy_target):
                    target_vec = _policy_target_to_vector(policy_target, n).to(device)
                    log_probs = F.log_softmax(sample_logits, dim=-1)
                    policy_loss = -(target_vec * log_probs).sum()
                else:
                    best_action = max(policy_target, key=lambda x: x[1])[0]
                    valid_action = [idx for idx in best_action if 0 <= idx < n]
                    if not valid_action:
                        continue
                    log_prob, _entropy = _sequential_log_prob(sample_logits, valid_action)
                    policy_loss = -log_prob
                policy_losses.append(policy_loss)
                used_indices.append(i)

            if not policy_losses:
                continue

            used_tensor = torch.tensor(used_indices, dtype=torch.long, device=device)
            policy_loss = torch.stack(policy_losses).mean()
            outcome_tensor = torch.tensor(outcomes, dtype=value.dtype, device=device)
            value_loss = F.mse_loss(
                value.squeeze(-1).index_select(0, used_tensor),
                outcome_tensor.index_select(0, used_tensor),
            )
            loss = policy_loss + value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(used_indices)
            total_policy_loss += policy_loss.item() * len(used_indices)
            total_value_loss += value_loss.item() * len(used_indices)
            count += len(used_indices)
        denom = max(1, count)
        print(
            f"[MCTS-train] epoch={epoch} avg_loss={total_loss/denom:.4f} "
            f"avg_policy_loss={total_policy_loss/denom:.4f} avg_value_loss={total_value_loss/denom:.4f} "
            f"samples_used={count}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=64, help="1意思決定あたりのMCTSシミュレーション数")
    parser.add_argument("--candidates", type=int, default=4, help="評価する候補手の上限数")
    parser.add_argument("--min_candidates", type=int, default=1, help="動的候補数の下限")
    parser.add_argument(
        "--dynamic_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="policyエントロピーに応じて候補手数を動的に決める",
    )
    parser.add_argument("--max_steps", type=int, default=300, help="1試合あたりの最大ステップ数")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="", help="初期重み(BCやPPOのチェックポイント)")
    parser.add_argument("--out", type=str, default="models/mcts_gen1.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vs_opponents",
        type=str,
        default="",
        help="カンマ区切りの対戦相手名(train_ppo.FIXED_OPPONENTSのキー、例: crustle,abomasnow)",
    )
    parser.add_argument("--vs_opponent_games", type=int, default=50, help="各対戦相手ごとの1世代あたりの対戦数")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, min(args.workers, 4)))

    my_deck = read_deck("deck.csv")

    net = PTCGNet()
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ckpt["model"], strict=False)
        print(f"[MCTS] resumed weights from {args.resume}")

    net.to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_root, out_ext = os.path.splitext(args.out)
    for gen in range(args.generations):
        print(f"[MCTS] === generation {gen} ===")
        print(f"[MCTS] collecting {args.games} self-play games with {args.workers} workers...")
        samples = collect_self_play(
            net,
            my_deck,
            args.games,
            args.workers,
            args.simulations,
            args.candidates,
            args.min_candidates,
            args.dynamic_candidates,
            args.max_steps,
            args.temperature,
        )
        if args.vs_opponents:
            opponent_names = [n.strip() for n in args.vs_opponents.split(",") if n.strip()]
            opponent_specs = []
            for name in opponent_names:
                opp_fn, deck_filename = FIXED_OPPONENTS[name]
                opp_deck = my_deck if deck_filename is None else read_deck(os.path.join(AGENT_DIR, deck_filename))
                opponent_specs.append((name, opp_fn, opp_deck))
            samples += collect_vs_opponents(
                net,
                my_deck,
                opponent_specs,
                args.vs_opponent_games,
                args.workers,
                args.simulations,
                args.candidates,
                args.min_candidates,
                args.dynamic_candidates,
                args.max_steps,
                args.temperature,
            )
        train(samples, net, optimizer, args.epochs, args.device, args.value_coef, args.batch_size)

        ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "episode": gen}
        gen_out = f"{out_root}_gen{gen}{out_ext}"
        torch.save(ckpt, gen_out)
        torch.save(ckpt, args.out)
        print(f"[MCTS] saved checkpoint to {gen_out}")
        print(f"[MCTS] saved latest checkpoint to {args.out}")


if __name__ == "__main__":
    main()
