#!/usr/bin/env python3
"""
対戦ビジュアライザ

使い方:
    python tools/visualize_battle.py              # セルフプレイ1戦
    python tools/visualize_battle.py --turns 20   # 最大20ターン表示
    python tools/visualize_battle.py --delay 0.5  # ターン間0.5秒待機
    python tools/visualize_battle.py --html       # HTMLファイルに出力

デッキは agent/deck.csv を使用。
"""

import sys, os, time, argparse, json, re

# cg モジュールのパスを追加
CG_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "sample_submission")
sys.path.insert(0, CG_BASE)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, LogType, EnergyType


# ── カード名 / エネルギー名 ルックアップ ────────────────────────

def _build_card_names() -> dict[int, str]:
    cards = all_card_data()
    return {c.cardId: c.name for c in cards}


ENERGY_SYMBOLS = {
    EnergyType.COLORLESS: "C",
    EnergyType.GRASS:     "G",
    EnergyType.FIRE:      "F",
    EnergyType.WATER:     "W",
    EnergyType.LIGHTNING: "L",
    EnergyType.PSYCHIC:   "P",
    EnergyType.FIGHTING:  "Fi",
    EnergyType.DARKNESS:  "D",
    EnergyType.METAL:     "M",
    EnergyType.DRAGON:    "N",
    EnergyType.RAINBOW:   "★",
}


def _energy_str(energies: list) -> str:
    if not energies:
        return ""
    counts: dict[str, int] = {}
    for e in energies:
        s = ENERGY_SYMBOLS.get(e, "?")
        counts[s] = counts.get(s, 0) + 1
    parts = []
    for sym, cnt in counts.items():
        parts.append(f"[{sym}]×{cnt}" if cnt > 1 else f"[{sym}]")
    return "".join(parts)


# ── デッキ読み込み ───────────────────────────────────────────────

def read_deck(path: str) -> list[int]:
    with open(path) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


# ── エージェント（ヒューリスティック） ─────────────────────────

AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")
sys.path.insert(0, AGENT_DIR)
import dragapult_agent as _da


def run_agent(obs_dict: dict, player_idx: int, deck: list[int]) -> list[int]:
    if obs_dict.get("select") is None:
        return deck
    # yourIndex をそのプレイヤー視点に一時書き換え
    import copy
    patched = copy.deepcopy(obs_dict)
    if patched.get("current"):
        patched["current"]["yourIndex"] = player_idx
    return _da.agent(patched)


# ── ターン状態の表示 ─────────────────────────────────────────────

WIDTH = 70

def _bar(hp: int, max_hp: int, width: int = 12) -> str:
    if max_hp == 0:
        return "[" + "?" * width + "]"
    filled = round(hp / max_hp * width)
    ratio  = hp / max_hp
    if ratio > 0.5:
        color = "\033[32m"   # green
    elif ratio > 0.25:
        color = "\033[33m"   # yellow
    else:
        color = "\033[31m"   # red
    reset = "\033[0m"
    return f"{color}[{'█' * filled}{'░' * (width - filled)}]{reset}"


def _pokemon_line(pk: dict, card_names: dict, prefix: str = "  ") -> str:
    if pk is None:
        return f"{prefix}(none)"
    cid   = pk.get("id", 0)
    name  = card_names.get(cid, f"#{cid}")
    hp    = pk.get("hp", 0)
    maxhp = pk.get("maxHp", hp)
    bar   = _bar(hp, maxhp)
    en    = _energy_str(pk.get("energies", []) or [])
    conds = []
    for sc in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if pk.get(sc):
            conds.append(sc[:3].upper())
    cond_str = " " + " ".join(conds) if conds else ""
    return f"{prefix}{name:<22} {bar} {hp:>3}/{maxhp:<3}  {en}{cond_str}"


def _player_block(state: dict, pidx: int, card_names: dict, is_you: bool) -> list[str]:
    players = state.get("players", [{}, {}])
    p = players[pidx] if pidx < len(players) else {}

    label = ("▶YOU " if is_you else "  OPP") + f" [P{pidx}]"
    prizes = len(p.get("prize", []) or [])
    hand   = p.get("handCount", len(p.get("hand") or []))
    deck   = p.get("deckCount", "?")

    lines = [
        f"  {label}   Prizes:{prizes}  Hand:{hand}  Deck:{deck}",
    ]

    active_list = p.get("active", []) or []
    active = active_list[0] if active_list else None
    lines.append(_pokemon_line(active, card_names, prefix="  Active: "))

    bench = p.get("bench", []) or []
    if bench:
        lines.append("  Bench:")
        for bk in bench:
            lines.append(_pokemon_line(bk, card_names, prefix="    "))
    return lines


LOG_TYPE_MAP = {
    LogType.TURN_START:  "Turn start",
    LogType.TURN_END:    "Turn end",
    LogType.DRAW:        "Draw",
    LogType.MOVE_CARD:   "Card moved",
    LogType.ATTACK:      "Attack",
    LogType.HP_CHANGE:   "HP change",
    LogType.PLAY:        "Play card",
    LogType.EVOLVE:      "Evolve",
    LogType.SWITCH:      "Switch",
    LogType.RESULT:      "Result",
}

RESULT_REASON = {1: "0 prizes", 2: "deck out", 3: "no active", 4: "card effect"}


def _format_logs(logs: list[dict], card_names: dict, last_n: int = 8) -> list[str]:
    lines = []
    for log in logs[-last_n:]:
        lt  = log.get("type", -1)
        pid = log.get("playerIndex", "?")

        if lt == LogType.ATTACK:
            cid  = log.get("cardId", 0)
            name = card_names.get(cid, f"#{cid}")
            lines.append(f"  P{pid} {name} attacked")

        elif lt == LogType.HP_CHANGE:
            cid   = log.get("cardId", 0)
            name  = card_names.get(cid, f"#{cid}")
            val   = log.get("value", 0)
            sign  = "+" if val > 0 else ""
            dc    = " (damage counter)" if log.get("putDamageCounter") else ""
            lines.append(f"  P{pid} {name} HP {sign}{val}{dc}")

        elif lt == LogType.EVOLVE:
            before = card_names.get(log.get("cardIdBefore", 0), "?")
            after  = card_names.get(log.get("cardIdAfter",  0), "?")
            lines.append(f"  P{pid} evolved {before} → {after}")

        elif lt == LogType.PLAY:
            cid  = log.get("cardId", 0)
            name = card_names.get(cid, f"#{cid}")
            lines.append(f"  P{pid} played {name}")

        elif lt == LogType.RESULT:
            winner = log.get("result", -1)
            reason = RESULT_REASON.get(log.get("reason"), "unknown")
            lines.append(f"  *** RESULT: P{winner} wins ({reason}) ***")

        elif lt == LogType.TURN_START:
            lines.append(f"  --- Turn start (P{pid}) ---")

    return lines


def print_state(obs: dict, card_names: dict, step: int, delay: float):
    state = obs.get("current")
    logs  = obs.get("logs", [])
    sel   = obs.get("select")

    if state is None:
        return

    turn       = state.get("turn", 0)
    your_idx   = state.get("yourIndex", 0)
    result     = state.get("result", -1)
    ctx        = sel.get("context") if sel else None

    print("\033[2J\033[H", end="")  # clear screen

    # ヘッダ
    result_str = "ONGOING" if result == -1 else f"P{result} WINS"
    print("═" * WIDTH)
    print(f"  Turn {turn:>3}  │  Deciding: P{your_idx}  │  {result_str}  │  Step {step}")
    print("═" * WIDTH)

    # 対戦ボード(相手が上、自分が下)
    opp_idx = 1 - your_idx
    print()
    for line in _player_block(state, opp_idx, card_names, is_you=False):
        print(line)
    print()
    print("  " + "─" * (WIDTH - 4))
    print()
    for line in _player_block(state, your_idx, card_names, is_you=True):
        print(line)
    print()

    # ログ
    if logs:
        print("  Recent events:")
        for line in _format_logs(logs, card_names):
            print(line)
        print()

    # 現在の選択コンテキスト
    if sel:
        ctx_name = ctx if isinstance(ctx, str) else str(ctx)
        opts_cnt = len(sel.get("option", []) or [])
        print(f"  → Select context: {ctx_name}  options: {opts_cnt}")

    if delay > 0:
        time.sleep(delay)


# ── HTML 出力 ───────────────────────────────────────────────────

def state_to_html_block(obs: dict, card_names: dict, step: int) -> str:
    state = obs.get("current")
    logs  = obs.get("logs", [])
    sel   = obs.get("select")

    if state is None:
        return ""

    turn       = state.get("turn", 0)
    your_idx   = state.get("yourIndex", 0)
    result     = state.get("result", -1)
    result_str = "ONGOING" if result == -1 else f"P{result} WINS"
    players    = state.get("players", [{}, {}])

    def player_html(pidx: int, is_you: bool) -> str:
        p      = players[pidx] if pidx < len(players) else {}
        label  = f"{'▶ YOU' if is_you else 'OPP'} [P{pidx}]"
        prizes = len(p.get("prize", []) or [])
        hand   = p.get("handCount", len(p.get("hand") or []))
        deck   = p.get("deckCount", "?")

        active_list = p.get("active", []) or []
        active      = active_list[0] if active_list else None

        def pk_row(pk, is_active: bool = False) -> str:
            if pk is None:
                return "<tr><td colspan='4'>(none)</td></tr>"
            cid   = pk.get("id", 0)
            name  = card_names.get(cid, f"#{cid}")
            hp    = pk.get("hp", 0)
            maxhp = pk.get("maxHp", hp)
            ratio = hp / maxhp if maxhp else 0
            color = ("#4CAF50" if ratio > 0.5 else "#FF9800" if ratio > 0.25 else "#F44336")
            pct   = int(ratio * 100)
            en    = _energy_str(pk.get("energies", []) or [])
            bold  = " font-weight:bold;" if is_active else ""
            return (f"<tr><td style='{bold}'> {'★ ' if is_active else '  '}{name}</td>"
                    f"<td><div style='background:#ddd;border-radius:4px;height:12px;width:80px'>"
                    f"<div style='background:{color};height:12px;width:{pct}%;border-radius:4px'></div></div></td>"
                    f"<td>{hp}/{maxhp}</td><td>{en}</td></tr>")

        bench = p.get("bench", []) or []
        bench_rows = "".join(pk_row(b) for b in bench)

        bg = "#e8f5e9" if is_you else "#fce4ec"
        return f"""
        <div style='background:{bg};padding:8px;margin:4px;border-radius:6px'>
        <b>{label}</b>  Prizes:{prizes}  Hand:{hand}  Deck:{deck}
        <table style='width:100%;border-collapse:collapse;font-size:13px;margin-top:4px'>
        {pk_row(active, True)}{bench_rows}
        </table></div>"""

    opp_idx   = 1 - your_idx
    opp_html  = player_html(opp_idx, False)
    you_html  = player_html(your_idx, True)

    log_items = ""
    for line in _format_logs(logs, card_names, last_n=6):
        clean = line.strip()
        if clean:
            log_items += f"<li>{clean}</li>"

    ctx = sel.get("context") if sel else None
    ctx_str = f"<p>→ <em>Select context:</em> {ctx}  options: {len(sel.get('option',[]))}</p>" if sel else ""

    return f"""
<div class='step' id='step{step}'>
  <div class='header'>Turn {turn} &nbsp;│&nbsp; P{your_idx} deciding &nbsp;│&nbsp; {result_str} &nbsp;│&nbsp; Step {step}</div>
  {opp_html}
  <hr style='margin:4px'>
  {you_html}
  {'<ul style="font-size:12px;margin:4px">' + log_items + '</ul>' if log_items else ''}
  {ctx_str}
</div>"""


HTML_STYLE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>PTCG Battle Replay</title>
<style>
body {{ font-family: monospace; font-size: 14px; background: #f5f5f5; }}
.step {{ background: white; border: 1px solid #ccc; border-radius: 8px;
         padding: 12px; margin: 12px auto; max-width: 700px; }}
.header {{ font-weight: bold; font-size: 15px; border-bottom: 2px solid #333;
           padding-bottom: 6px; margin-bottom: 8px; }}
table {{ width: 100%; }}
td {{ padding: 2px 6px; }}
hr {{ border-color: #bbb; }}
ul {{ list-style: disc; padding-left: 20px; }}
</style>
</head><body>
<h2 style='text-align:center'>PTCG Battle Replay</h2>
{blocks}
</body></html>"""


# ── メインループ ─────────────────────────────────────────────────

def run(deck: list[int], max_steps: int, delay: float,
        html_path: str | None, log_file: str | None):

    card_names = _build_card_names()
    print("Card database loaded:", len(card_names), "cards")
    print("Starting battle ...\n")

    obs, start_data = battle_start(deck, deck[:])  # セルフプレイ
    if obs is None:
        print("Battle failed to start:", start_data)
        return

    html_blocks = []
    jsonl_lines = []

    step = 0
    result = -1

    # デッキ選択 (select is None)
    for pidx in range(2):
        fake_obs = {"select": None, "logs": [], "current": None}
        action = run_agent(fake_obs, pidx, deck)
        # デッキはすでに battle_start で渡した

    while step < max_steps:
        step += 1

        current = obs.get("current") if obs else None
        if current:
            result = current.get("result", -1)

        if html_path:
            html_blocks.append(state_to_html_block(obs, card_names, step))
        else:
            print_state(obs, card_names, step, delay)

        if log_file:
            jsonl_lines.append(json.dumps(obs, ensure_ascii=False))

        if result != -1:
            break

        sel = obs.get("select")
        if sel is None:
            break

        your_idx = current.get("yourIndex", 0) if current else 0
        action   = run_agent(obs, your_idx, deck)

        if not action:
            action = [0]

        try:
            obs = battle_select(action)
        except Exception as e:
            print(f"[Error at step {step}] battle_select({action}): {e}")
            break

    battle_finish()

    # 結果サマリー
    winner = result
    if html_path:
        html = HTML_STYLE.format(blocks="\n".join(html_blocks))
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nHTML replay saved to: {html_path}")
    else:
        print("\n" + "═" * WIDTH)
        if winner == -1:
            print(f"  Battle ended after {step} steps (max_steps reached)")
        else:
            print(f"  P{winner} WINS after {step} steps!")
        print("═" * WIDTH)

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))
        print(f"Battle log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="PTCG Battle Visualizer")
    parser.add_argument("--turns",  type=int,   default=500,  help="Max steps (default 500)")
    parser.add_argument("--delay",  type=float, default=0.3,  help="Seconds between turns (default 0.3)")
    parser.add_argument("--html",   type=str,   default=None, help="Output HTML replay file")
    parser.add_argument("--log",    type=str,   default=None, help="Output JSONL log file")
    parser.add_argument("--deck",   type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "agent", "deck.csv"),
                        help="Path to deck.csv")
    args = parser.parse_args()

    deck = read_deck(args.deck)
    print(f"Deck: {len(deck)} cards loaded from {args.deck}")

    run(deck, args.turns, args.delay, args.html, args.log)


if __name__ == "__main__":
    main()
