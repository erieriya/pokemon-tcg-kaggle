#!/usr/bin/env python3
"""
対戦を実行して JSON ログを保存する。

使い方:
    python tools/run_battle.py                          # dragapult_agent同士、battle_logs/battle_YYYYMMDD_HHMMSS.json
    python tools/run_battle.py --out my_game.json       # 出力先指定
    python tools/run_battle.py --rl-model agent/models/model_final.pt  # 学習済みRLエージェント同士の自己対戦
    python tools/run_battle.py --rl-model agent/models/model_final.pt --opponent crustle
        # 学習済みRLエージェント(Dragapult ex) vs 固定対戦相手(train_ppo.pyのFIXED_OPPONENTSと同じ定義)
"""

import random
import sys, os, json, argparse, copy
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_BASE    = os.path.join(BASE_DIR, "data", "sample_submission")
AGENT_DIR  = os.path.join(BASE_DIR, "agent")
LOG_DIR    = os.path.join(BASE_DIR, "battle_logs")

sys.path.insert(0, CG_BASE)
sys.path.insert(0, AGENT_DIR)

from cg.game import battle_start, battle_select, battle_finish
from cg.api  import all_card_data, all_attack
import dragapult_agent as _agent_mod
import lucario_v1_agent
import lucario_v2_agent
import crustle_agent
import iono_agent
import abomasnow_agent


def build_card_db() -> tuple[dict, dict]:
    cards    = all_card_data()
    attacks  = all_attack()
    card_db  = {c.cardId: c for c in cards}
    attack_db = {a.attackId: a for a in attacks}
    return card_db, attack_db


def read_deck(path: str | None = None) -> list:
    path = path or os.path.join(AGENT_DIR, "deck.csv")
    with open(path) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


# train_ppo.pyのFIXED_OPPONENTSと同じ定義(名前 -> (agent関数 obs_dict->list[int], デッキファイル名))。
# "mirror"は特別扱いで、player0と同じエージェント・デッキを使う(従来の自己対戦と同じ動作)。
def _random_opponent(obs_dict: dict) -> list[int]:
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


FIXED_OPPONENTS = {
    "random": (_random_opponent, None),
    "dragapult": (_agent_mod.agent, "deck.csv"),
    "lucario_v1": (lucario_v1_agent.agent, "deck_lucario_v1.csv"),
    "lucario_v2": (lucario_v2_agent.agent, "deck_lucario_v2.csv"),
    "crustle": (crustle_agent.agent, "deck_crustle.csv"),
    "iono": (iono_agent.agent, "deck_iono.csv"),
    "abomasnow": (abomasnow_agent.agent, "deck_abomasnow.csv"),
}


def make_call_agent(rl_model_path: str | None):
    """rl_model_pathが指定されていれば学習済みRLエージェント、なければdragapult_agentを使う関数を返す。"""
    if rl_model_path:
        from rl_agent import RLAgent
        rl = RLAgent(model_path=rl_model_path)

        def call_agent(obs_dict: dict, player_idx: int, deck: list) -> list:
            if obs_dict.get("select") is None:
                return deck
            patched = copy.deepcopy(obs_dict)
            if patched.get("current"):
                patched["current"]["yourIndex"] = player_idx
            return rl(patched)

        return call_agent

    def call_agent(obs_dict: dict, player_idx: int, deck: list) -> list:
        if obs_dict.get("select") is None:
            return deck
        patched = copy.deepcopy(obs_dict)
        if patched.get("current"):
            patched["current"]["yourIndex"] = player_idx
        return _agent_mod.agent(patched)

    return call_agent


def _sanitize_action(action, sel: dict, n: int) -> list[int]:
    """エージェントの返り値を検証し、不正ならエンジンが受理する合法手にフォールバックする
    (agent/train_ppo.pyの_sanitize_opponent_actionと同じロジック。型・範囲だけでなく
    重複の有無とminCount/maxCountの個数も検証する)。"""
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


def make_opponent_call_agent(name: str):
    """固定対戦相手(FIXED_OPPONENTS)を player1 用の call_agent(obs_dict, player_idx, deck) に変換する。"""
    agent_fn, _ = FIXED_OPPONENTS[name]

    def call_agent(obs_dict: dict, player_idx: int, deck: list) -> list:
        if obs_dict.get("select") is None:
            return deck
        patched = copy.deepcopy(obs_dict)
        if patched.get("current"):
            patched["current"]["yourIndex"] = player_idx
        action = agent_fn(patched)
        sel = obs_dict.get("select") or {}
        n = len(sel.get("option") or [])
        return _sanitize_action(action, sel, n)

    return call_agent


def serialize_pokemon(pk: dict, card_db: dict) -> dict | None:
    if pk is None:
        return None
    cid = pk.get("id", 0)
    return {
        "id":        cid,
        "hp":        pk.get("hp", 0),
        "maxHp":     pk.get("maxHp", pk.get("hp", 0)),
        "energies":  [int(e) for e in (pk.get("energies") or [])],
        "tools":     list(pk.get("tools") or []),
        "poisoned":  bool(pk.get("poisoned")),
        "burned":    bool(pk.get("burned")),
        "asleep":    bool(pk.get("asleep")),
        "paralyzed": bool(pk.get("paralyzed")),
        "confused":  bool(pk.get("confused")),
        "isNew":     bool(pk.get("appearThisTurn")),
    }


def serialize_player(p: dict, card_db: dict) -> dict:
    if not p:
        return {}
    active_list  = p.get("active") or []
    bench_list   = p.get("bench")  or []
    hand_raw     = p.get("hand")
    prize_list   = p.get("prize")  or []
    discard_list = p.get("discard") or []

    hand_cards = None
    if hand_raw is not None:
        hand_cards = []
        for c in hand_raw:
            if c is None:
                continue
            cid = c.get("id", 0) if isinstance(c, dict) else 0
            hand_cards.append(cid)

    return {
        "active":       serialize_pokemon(active_list[0] if active_list else None, card_db),
        "bench":        [serialize_pokemon(b, card_db) for b in bench_list if b is not None],
        "hand":         hand_cards,
        "handCount":    p.get("handCount", len(hand_raw) if hand_raw else 0),
        "deckCount":    p.get("deckCount", 0),
        "prizeCount":   len(prize_list),
        "discardCount": len(discard_list),
    }


def serialize_logs(logs: list) -> list:
    return logs or []


def run_battle(
    card_db: dict,
    call_agent0,
    deck0: list,
    call_agent1,
    deck1: list,
    agent_name: str,
    max_steps: int = 2000,
) -> dict:
    """player0 = call_agent0/deck0, player1 = call_agent1/deck1 で1戦実行する。

    ミラー戦(従来の自己対戦)はcall_agent0==call_agent1, deck0==deck1を渡せばよい。
    """
    call_agents = (call_agent0, call_agent1)
    decks = (deck0, deck1)
    obs_dict, _ = battle_start(deck0, deck1)

    states   = []
    step_num = 0

    while step_num < max_steps:
        state    = obs_dict.get("current") or {}
        sel      = obs_dict.get("select")
        logs     = obs_dict.get("logs") or []
        turn     = state.get("turn", 0)
        your_idx = state.get("yourIndex", 0)
        result   = state.get("result", -1)
        players  = state.get("players") or [{}, {}]
        ctx      = int(sel.get("context", -1)) if sel else None

        step_state = {
            "step":    step_num,
            "turn":    turn,
            "player":  your_idx,
            "result":  result,
            "context": ctx,
            "players": [
                serialize_player(players[0] if players else {}, card_db),
                serialize_player(players[1] if len(players) > 1 else {}, card_db),
            ],
            "logs":    serialize_logs(logs),
            "action":  None,
        }

        if result != -1 or sel is None:
            states.append(step_state)
            break

        action = call_agents[your_idx](obs_dict, your_idx, decks[your_idx])
        step_state["action"] = action
        states.append(step_state)

        obs_dict = battle_select(action)
        step_num += 1

        next_state  = obs_dict.get("current") or {}
        next_result = next_state.get("result", -1)
        next_sel    = obs_dict.get("select")

        if next_result != -1 or next_sel is None:
            logs2    = obs_dict.get("logs") or []
            players2 = next_state.get("players") or [{}, {}]
            states.append({
                "step":    step_num,
                "turn":    next_state.get("turn", turn),
                "player":  next_state.get("yourIndex", your_idx),
                "result":  next_result,
                "context": None,
                "players": [
                    serialize_player(players2[0] if players2 else {}, card_db),
                    serialize_player(players2[1] if len(players2) > 1 else {}, card_db),
                ],
                "logs":    serialize_logs(logs2),
                "action":  None,
            })
            break

    battle_finish()
    return {
        "metadata": {
            "date":        datetime.now().isoformat(),
            "agent":       agent_name,
            "total_steps": len(states),
            "total_turns": states[-1]["turn"] if states else 0,
            "result":      states[-1]["result"] if states else -1,
        },
        "states": states,
    }


def main():
    parser = argparse.ArgumentParser(description="対戦を実行して JSON ログを保存")
    parser.add_argument("--out", default=None, help="出力先 JSON ファイルパス")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--rl-model", default=None,
                         help="指定すると学習済みRLエージェント同士の自己対戦になる (例: agent/models/model_final.pt)")
    parser.add_argument(
        "--opponent", default="mirror", choices=["mirror", *FIXED_OPPONENTS.keys()],
        help="player1側の対戦相手。mirror(既定)はplayer0と同じエージェント・デッキでの自己対戦。"
             "それ以外はtrain_ppo.pyのFIXED_OPPONENTSと同じ固定対戦相手(専用デッキ込み)。",
    )
    parser.add_argument("--player0-deck", default=None, help="player0用デッキcsv(既定: agent/deck.csv)")
    args = parser.parse_args()

    print("カードデータ読み込み中...")
    card_db, _ = build_card_db()
    print(f"  {len(card_db)} 枚")

    deck0 = read_deck(args.player0_deck)
    label0 = f"rl_agent({args.rl_model})" if args.rl_model else "dragapult_agent"
    call_agent0 = make_call_agent(args.rl_model)

    if args.opponent == "mirror":
        call_agent1, deck1, label1 = call_agent0, deck0, label0
    else:
        call_agent1 = make_opponent_call_agent(args.opponent)
        _, deck_filename = FIXED_OPPONENTS[args.opponent]
        deck1 = read_deck(os.path.join(AGENT_DIR, deck_filename)) if deck_filename else deck0
        label1 = args.opponent

    agent_name = label0 if args.opponent == "mirror" else f"{label0} vs {label1}"
    print(f"対戦実行中... ({agent_name})")
    battle_log = run_battle(card_db, call_agent0, deck0, call_agent1, deck1, agent_name, max_steps=args.max_steps)
    meta = battle_log["metadata"]
    winner = meta["result"]
    winner_label = {0: label0, 1: label1}.get(winner, "不明")
    print(f"  完了: {meta['total_steps']} ステップ  "
          f"Turn {meta['total_turns']}  "
          f"結果: {'P' + str(winner) + ' (' + winner_label + ') 勝利' if winner != -1 else '不明'}")

    if args.out:
        out_path = args.out
    else:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(LOG_DIR, f"battle_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(battle_log, f, ensure_ascii=False, indent=None)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"保存: {out_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
