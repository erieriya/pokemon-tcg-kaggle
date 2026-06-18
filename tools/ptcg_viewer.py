#!/usr/bin/env python3
"""
PTCG スタイル 対戦リプレイ HTML ジェネレータ

使い方:
    # 新しい対戦を実行して HTML 生成
    python tools/ptcg_viewer.py

    # 保存済み JSON から HTML 生成
    python tools/ptcg_viewer.py --from-json battle_logs/battle_20260618_123456.json

    # 出力先とコピー先を指定
    python tools/ptcg_viewer.py --out game.html --copy-to-windows
"""

import sys, os, json, argparse, copy, shutil
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_BASE   = os.path.join(BASE_DIR, "data", "sample_submission")
AGENT_DIR = os.path.join(BASE_DIR, "agent")
LOG_DIR   = os.path.join(BASE_DIR, "battle_logs")

sys.path.insert(0, CG_BASE)
sys.path.insert(0, AGENT_DIR)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, EnergyType, CardType, LogType
import dragapult_agent as _agent_mod


# ── 日本語カード名 ───────────────────────────────────────────────

JP_NAMES: dict[str, str] = {
    # ポケモン
    "Dreepy":               "ドラメシヤ",
    "Drakloak":             "ドロンチ",
    "Dragapult ex":         "ドラパルトex",
    "Duskull":              "ヨマワル",
    "Dusclops":             "ダスクロップス",
    "Dusknoir":             "ヨノワール",
    "Fezandipiti ex":       "フェザーナex",
    "Budew":                "スボミー",
    "Roselia":              "ロゼリア",
    "Roserade":             "ロズレイド",
    "Scrafty":              "ズルズキン",
    "Hippopotas":           "カバルドン(幼)",
    "Hippowdon":            "カバルドン",
    "Cinderace ex":         "エースバーンex",
    "Lapras ex":            "ラプラスex",
    # グッズ
    "Ultra Ball":           "ハイパーボール",
    "Rare Candy":           "ふしぎなアメ",
    "Buddy-Buddy Poffin":   "なかよしポフィン",
    "Dusk Ball":            "ダークボール",
    "Bug Catching Set":     "むしとりセット",
    "Pokégear 3.0":         "ポケモンギア3.0",
    "Roto-Stick":           "ロトスティック",
    "Hole-Digging Shovel":  "あなほりスコップ",
    # ACE SPEC
    "Unfair Stamp":         "アンフェアスタンプ",
    # サポート
    "Hilda":                "トウコ",
    "Crispin":              "リキ",
    "Boss's Orders":        "ボスの指令",
    "Judge":                "ジャッジマン",
    "Billy & O'Nare":       "ビリー＆オナレ",
    "Perrin":               "ベル",
    # エネルギー
    "Basic {G} Energy":     "基本草エネルギー",
    "Basic {R} Energy":     "基本炎エネルギー",
    "Basic {W} Energy":     "基本水エネルギー",
    "Basic {L} Energy":     "基本雷エネルギー",
    "Basic {P} Energy":     "基本超エネルギー",
    "Basic {F} Energy":     "基本闘エネルギー",
    "Basic {D} Energy":     "基本悪エネルギー",
    "Basic {M} Energy":     "基本鋼エネルギー",
    "Basic {N} Energy":     "基本竜エネルギー",
}


def jp(name: str) -> str:
    return JP_NAMES.get(name, name)


# ── エネルギー表示 ───────────────────────────────────────────────

ENERGY_COLOR: dict[int, str] = {
    int(EnergyType.COLORLESS): "#A8A77A",
    int(EnergyType.GRASS):     "#7AC74C",
    int(EnergyType.FIRE):      "#EE8130",
    int(EnergyType.WATER):     "#6390F0",
    int(EnergyType.LIGHTNING): "#F7D02C",
    int(EnergyType.PSYCHIC):   "#F95587",
    int(EnergyType.FIGHTING):  "#C22E28",
    int(EnergyType.DARKNESS):  "#705848",
    int(EnergyType.METAL):     "#B7B7CE",
    int(EnergyType.DRAGON):    "#6F35FC",
    int(EnergyType.RAINBOW):   "#888",
}

ENERGY_LABEL: dict[int, str] = {
    int(EnergyType.COLORLESS): "無",
    int(EnergyType.GRASS):     "草",
    int(EnergyType.FIRE):      "炎",
    int(EnergyType.WATER):     "水",
    int(EnergyType.LIGHTNING): "雷",
    int(EnergyType.PSYCHIC):   "超",
    int(EnergyType.FIGHTING):  "闘",
    int(EnergyType.DARKNESS):  "悪",
    int(EnergyType.METAL):     "鋼",
    int(EnergyType.DRAGON):    "竜",
    int(EnergyType.RAINBOW):   "★",
}

ENERGY_NAME_JP: dict[int, str] = {
    int(EnergyType.COLORLESS): "無色",
    int(EnergyType.GRASS):     "草",
    int(EnergyType.FIRE):      "炎",
    int(EnergyType.WATER):     "水",
    int(EnergyType.LIGHTNING): "雷",
    int(EnergyType.PSYCHIC):   "超",
    int(EnergyType.FIGHTING):  "闘",
    int(EnergyType.DARKNESS):  "悪",
    int(EnergyType.METAL):     "鋼",
    int(EnergyType.DRAGON):    "竜",
    int(EnergyType.RAINBOW):   "レインボー",
}

STAGE_JP: dict = {
    "basic":  "たね",
    "stage1": "1進化",
    "stage2": "2進化",
}


# ── カードデータ構築 ─────────────────────────────────────────────

def build_card_info(card_db: dict, attack_db: dict) -> dict:
    """
    HTML に埋め込む CARD_INFO を構築する。
    {cardId: {name, jpName, hp, stage, energyType, retreatCost,
              weakness, resistance, attacks:[{name,damage,energies,text}],
              abilities:[{name,text}], cardType}}
    """
    info = {}
    for cid, c in card_db.items():
        stage = ("basic"  if c.basic  else
                 "stage1" if c.stage1 else
                 "stage2" if c.stage2 else "basic")

        attacks = []
        for aid in (c.attacks or []):
            a = attack_db.get(aid)
            if a:
                attacks.append({
                    "name":     a.name,
                    "damage":   a.damage,
                    "energies": [int(e) for e in (a.energies or [])],
                    "text":     a.text or "",
                })

        abilities = []
        for sk in (c.skills or []):
            abilities.append({
                "name": sk.name,
                "text": sk.text or "",
            })

        info[cid] = {
            "name":        c.name,
            "jpName":      jp(c.name),
            "hp":          c.hp,
            "stage":       stage,
            "energyType":  int(c.energyType) if c.energyType is not None else -1,
            "retreatCost": c.retreatCost or 0,
            "weakness":    int(c.weakness)    if c.weakness    is not None else -1,
            "resistance":  int(c.resistance)  if c.resistance  is not None else -1,
            "ex":          bool(c.ex),
            "tera":        bool(c.tera),
            "aceSpec":     bool(c.aceSpec),
            "attacks":     attacks,
            "abilities":   abilities,
            "cardType":    int(c.cardType),
            "evolvesFrom": c.evolvesFrom or "",
        }
    return info


# ── ログシリアライズ ─────────────────────────────────────────────

LOG_TYPE_MAP = {
    int(LogType.ATTACK):     "attack",
    int(LogType.HP_CHANGE):  "hp",
    int(LogType.EVOLVE):     "evolve",
    int(LogType.PLAY):       "play",
    int(LogType.RESULT):     "result",
    int(LogType.TURN_START): "turn",
    int(LogType.TURN_END):   "turn",
    int(LogType.DRAW):       "draw",
}

RESULT_REASON = {
    1: "サイドを取りきった",
    2: "デッキアウト",
    3: "バトル場が空",
    4: "カード効果",
}


def format_logs(logs: list, card_db: dict) -> list:
    out = []
    for log in (logs or []):
        lt  = int(log.get("type", -1))
        pid = log.get("playerIndex", "?")
        cls = LOG_TYPE_MAP.get(lt, "misc")

        if lt == int(LogType.ATTACK):
            cid  = log.get("cardId", 0)
            name = jp(card_db[cid].name) if cid in card_db else f"#{cid}"
            text = f"P{pid}の{name}が攻撃！"
        elif lt == int(LogType.HP_CHANGE):
            cid  = log.get("cardId", 0)
            name = jp(card_db[cid].name) if cid in card_db else f"#{cid}"
            val  = log.get("value", 0)
            sign = "+" if val > 0 else ""
            dc   = " (ダメカン)" if log.get("putDamageCounter") else ""
            text = f"P{pid} {name} HP {sign}{val}{dc}"
        elif lt == int(LogType.EVOLVE):
            bid  = log.get("cardIdBefore", 0)
            aid2 = log.get("cardIdAfter",  0)
            bn   = jp(card_db[bid].name) if bid in card_db else f"#{bid}"
            an   = jp(card_db[aid2].name) if aid2 in card_db else f"#{aid2}"
            text = f"P{pid} {bn}→{an} 進化！"
            cls  = "evolve"
        elif lt == int(LogType.PLAY):
            cid  = log.get("cardId", 0)
            name = jp(card_db[cid].name) if cid in card_db else f"#{cid}"
            text = f"P{pid}が{name}を使用"
        elif lt == int(LogType.RESULT):
            winner = log.get("result", -1)
            reason = RESULT_REASON.get(log.get("reason"), "不明")
            text   = f"★ P{winner}の勝利！({reason})"
        elif lt == int(LogType.TURN_START):
            text = f"── ターン開始 (P{pid}) ──"
        elif lt == int(LogType.TURN_END):
            text = f"── ターン終了 (P{pid}) ──"
        elif lt == int(LogType.DRAW):
            count = log.get("count", 1)
            text  = f"P{pid}が{count}枚ドロー"
        else:
            text = f"P{pid} [type={lt}]"
            cls  = "misc"

        out.append({"text": text, "cls": cls})
    return out


# ── 対戦状態シリアライズ ─────────────────────────────────────────

def serialize_pokemon(pk: dict) -> dict | None:
    if pk is None:
        return None
    return {
        "id":        pk.get("id", 0),
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


def serialize_player(p: dict) -> dict:
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
        "active":       serialize_pokemon(active_list[0] if active_list else None),
        "bench":        [serialize_pokemon(b) for b in bench_list if b is not None],
        "hand":         hand_cards,
        "handCount":    p.get("handCount", len(hand_raw) if hand_raw else 0),
        "deckCount":    p.get("deckCount", 0),
        "prizeCount":   len(prize_list),
        "discardCount": len(discard_list),
    }


def run_battle_inline(card_db: dict) -> list:
    deck = read_deck()
    obs_dict, _ = battle_start(deck, deck)

    states   = []
    step_num = 0

    while step_num < 2000:
        state    = obs_dict.get("current") or {}
        sel      = obs_dict.get("select")
        logs_raw = obs_dict.get("logs") or []
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
                serialize_player(players[0] if players else {}),
                serialize_player(players[1] if len(players) > 1 else {}),
            ],
            "logs":    format_logs(logs_raw, card_db),
            "action":  None,
        }

        if result != -1 or sel is None:
            states.append(step_state)
            break

        action = call_agent(obs_dict, your_idx, deck)
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
                    serialize_player(players2[0] if players2 else {}),
                    serialize_player(players2[1] if len(players2) > 1 else {}),
                ],
                "logs":    format_logs(logs2, card_db),
                "action":  None,
            })
            break

    battle_finish()
    return states


def load_json_battle(path: str, card_db: dict) -> list:
    """保存済み JSON バトルログを読み込み、ログを日本語フォーマットで再処理"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "states" in data:
        states = data["states"]
    else:
        states = data  # 古い形式 (states のリスト直接)

    # raw logs → formatted logs に変換
    for s in states:
        raw_logs = s.get("logs") or []
        if raw_logs and isinstance(raw_logs[0], dict) and "text" not in raw_logs[0]:
            s["logs"] = format_logs(raw_logs, card_db)
        # hand が id リストの場合はそのまま（HTML 側で処理）
    return states


def read_deck() -> list:
    path = os.path.join(AGENT_DIR, "deck.csv")
    with open(path) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


def call_agent(obs_dict: dict, player_idx: int, deck: list) -> list:
    if obs_dict.get("select") is None:
        return deck
    patched = copy.deepcopy(obs_dict)
    if patched.get("current"):
        patched["current"]["yourIndex"] = player_idx
    return _agent_mod.agent(patched)


# ── HTML テンプレート ────────────────────────────────────────────

def generate_html(states: list, card_info: dict) -> str:
    states_json    = json.dumps(states,    ensure_ascii=False)
    card_info_json = json.dumps(card_info, ensure_ascii=False)
    energy_color   = json.dumps({str(k): v for k, v in ENERGY_COLOR.items()})
    energy_label   = json.dumps({str(k): v for k, v in ENERGY_LABEL.items()})
    energy_name    = json.dumps({str(k): v for k, v in ENERGY_NAME_JP.items()})

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Pokemon TCG Battle Replay</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1b0e;color:#e8f5e9;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;min-height:100vh;display:flex;flex-direction:column;align-items:center}}

/* ヘッダ */
#header{{width:100%;background:linear-gradient(135deg,#1a3a1f,#0d2610);border-bottom:2px solid #2e7d32;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}}
#header h1{{font-size:16px;color:#ffd700;letter-spacing:1px}}
.info-badge{{background:#1b5e20;border:1px solid #43a047;border-radius:6px;padding:3px 10px}}
#result-badge.win0{{background:#0d47a1;border-color:#42a5f5;color:#90caf9;font-weight:bold}}
#result-badge.win1{{background:#b71c1c;border-color:#ef5350;color:#ffcdd2;font-weight:bold}}

/* ナビ */
#nav{{display:flex;gap:8px;align-items:center;background:#1b2e1d;padding:8px 16px;border-bottom:1px solid #2e7d32;width:100%;justify-content:center;flex-wrap:wrap}}
button{{background:#1b5e20;color:#a5d6a7;border:1px solid #43a047;border-radius:6px;padding:5px 14px;cursor:pointer;font-size:13px;transition:background 0.15s}}
button:hover{{background:#2e7d32;color:#e8f5e9}}
button:disabled{{opacity:0.4;cursor:not-allowed}}
#step-slider{{width:240px;accent-color:#43a047}}
#step-label{{color:#81c784;min-width:80px;text-align:center}}
#speed-select{{background:#1b5e20;color:#a5d6a7;border:1px solid #43a047;border-radius:4px;padding:4px 6px}}
#load-btn{{background:#0d47a1;border-color:#42a5f5;color:#90caf9}}
#load-btn:hover{{background:#1565c0}}

/* メインレイアウト */
#main{{display:flex;gap:12px;padding:12px;width:100%;max-width:1280px}}

/* ボード */
#board{{flex:1;background:radial-gradient(ellipse at center,#1a472a 0%,#0d2610 100%);border:2px solid #2e7d32;border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:8px;min-width:0}}
.player-side{{display:flex;align-items:center;gap:8px}}

/* プライズ */
.prize-zone{{display:flex;flex-direction:column;gap:4px;align-items:center;min-width:52px}}
.prize-label{{font-size:9px;color:#66bb6a;text-transform:uppercase;letter-spacing:1px}}
.prize-stack{{display:flex;flex-direction:column;gap:2px}}
.prize-card{{width:32px;height:22px;background:linear-gradient(135deg,#1565c0,#283593);border:1px solid #5c6bc0;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;color:#90caf9}}
.prize-count{{font-size:18px;font-weight:bold;color:#ffd700;text-shadow:0 0 6px #ffd700aa}}

/* フィールド */
.field-zone{{flex:1;display:flex;flex-direction:column;gap:6px;min-width:0}}
.bench-row{{display:flex;gap:5px;justify-content:center;flex-wrap:nowrap}}

/* デッキ */
.deck-zone{{display:flex;flex-direction:column;gap:6px;align-items:center;min-width:56px}}
.deck-pile{{width:48px;height:64px;border-radius:5px;border:1px solid #43a047;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:10px;gap:2px}}
.deck-pile.deck-back{{background:linear-gradient(135deg,#4a148c,#311b92);border-color:#7b1fa2;box-shadow:2px 2px 0 #1a0033}}
.deck-pile.discard-back{{background:linear-gradient(135deg,#424242,#212121);border-color:#616161}}
.deck-pile-label{{font-size:9px;color:#aaa;text-transform:uppercase}}
.deck-pile-count{{font-size:15px;font-weight:bold;color:#e0e0e0}}

/* 区切り */
.board-divider{{display:flex;align-items:center;gap:8px;padding:2px 0}}
.divider-line{{flex:1;height:1px;background:linear-gradient(to right,transparent,#43a047,transparent)}}
.divider-vs{{color:#ffd700;font-size:11px;font-weight:bold;opacity:0.7}}
.deciding-indicator{{text-align:center;font-size:10px;color:#ffd700;padding:2px 0;letter-spacing:1px}}

/* ポケモンカード */
.pokemon-card{{width:90px;min-height:114px;border-radius:7px;border:2px solid #43a047;background:linear-gradient(160deg,#1b3a1f 0%,#0d2010 100%);display:flex;flex-direction:column;align-items:center;padding:5px 4px 4px;gap:3px;position:relative;cursor:pointer;transition:transform 0.15s,box-shadow 0.15s;user-select:none}}
.pokemon-card.active-card{{border-color:#ffd700;box-shadow:0 0 12px #ffd70066;width:104px;min-height:128px}}
.pokemon-card.empty-slot{{border-style:dashed;border-color:#2e4030;background:rgba(0,0,0,0.2);opacity:0.4;cursor:default}}
.pokemon-card:hover:not(.empty-slot){{transform:translateY(-3px);box-shadow:0 6px 16px #00000066;z-index:5}}

/* カード内部 */
.card-header{{width:100%;display:flex;align-items:flex-start;justify-content:space-between;gap:2px}}
.card-name-jp{{font-size:10px;font-weight:bold;color:#c8e6c9;line-height:1.2;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.card-name-en{{font-size:7px;color:#558b5a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:100%}}
.card-ex{{font-size:8px;color:#ffb74d;font-weight:bold;flex-shrink:0}}
.card-new-badge{{background:#f57f17;color:#fff;font-size:7px;padding:1px 3px;border-radius:3px;font-weight:bold}}
.card-tera{{font-size:7px;color:#81d4fa;font-weight:bold}}

/* HP バー */
.hp-bar-container{{width:100%;height:6px;background:#0a1a0b;border-radius:3px;overflow:hidden}}
.hp-bar-fill{{height:100%;border-radius:3px;transition:width 0.3s}}
.hp-bar-fill.hp-high{{background:linear-gradient(to right,#2e7d32,#66bb6a)}}
.hp-bar-fill.hp-mid{{background:linear-gradient(to right,#f57f17,#ffb74d)}}
.hp-bar-fill.hp-low{{background:linear-gradient(to right,#b71c1c,#ef5350)}}
.hp-text{{font-size:9px;color:#a5d6a7;width:100%;text-align:right}}

/* エネルギー */
.energies{{display:flex;flex-wrap:wrap;gap:2px;justify-content:center}}
.energy-dot{{width:15px;height:15px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:7.5px;font-weight:bold;color:#fff;border:1px solid rgba(255,255,255,0.3);text-shadow:0 0 2px #000;flex-shrink:0}}

/* 状態異常 */
.status-badges{{display:flex;flex-wrap:wrap;gap:2px;justify-content:center}}
.status-badge{{font-size:7px;padding:1px 3px;border-radius:3px;font-weight:bold}}
.status-psn{{background:#7b1fa2;color:#e1bee7}}
.status-brn{{background:#bf360c;color:#ffccbc}}
.status-slp{{background:#37474f;color:#b0bec5}}
.status-par{{background:#f9a825;color:#fff}}
.status-cnf{{background:#4a148c;color:#e1bee7}}

/* ツール */
.tool-badge{{font-size:7px;background:#1565c0;color:#90caf9;padding:1px 3px;border-radius:2px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}

/* 手札 */
#hand-zone{{width:100%;max-width:1280px;padding:8px 12px;border-top:1px solid #2e7d32;background:rgba(0,0,0,0.3)}}
#hand-zone h3{{font-size:11px;color:#66bb6a;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px}}
#hand-cards{{display:flex;flex-wrap:wrap;gap:4px}}
.hand-card{{padding:3px 8px;border-radius:4px;font-size:11px;border:1px solid;cursor:pointer;transition:opacity 0.15s}}
.hand-card:hover{{opacity:0.8}}
.hand-card.type-0{{background:#1b5e20;border-color:#43a047;color:#c8e6c9}}
.hand-card.type-1,.hand-card.type-2{{background:#01579b;border-color:#0288d1;color:#81d4fa}}
.hand-card.type-3{{background:#4a148c;border-color:#7b1fa2;color:#ce93d8}}
.hand-card.type-4{{background:#1a237e;border-color:#3949ab;color:#9fa8da}}
.hand-card.type-5,.hand-card.type-6{{background:#e65100;border-color:#f57c00;color:#ffcc80}}

/* ログ */
#log-panel{{width:230px;flex-shrink:0;display:flex;flex-direction:column;gap:8px}}
.log-box{{background:#0a1a0b;border:1px solid #2e7d32;border-radius:8px;padding:8px;overflow-y:auto;max-height:440px}}
.log-box h3{{font-size:10px;color:#66bb6a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid #1b5e20;padding-bottom:4px}}
.log-entry{{padding:3px 0;font-size:11px;border-bottom:1px solid #0d2610;line-height:1.4}}
.log-entry.attack{{color:#ef9a9a}}
.log-entry.hp{{color:#ffcc80}}
.log-entry.evolve{{color:#a5d6a7}}
.log-entry.play{{color:#90caf9}}
.log-entry.result{{color:#ffd700;font-weight:bold}}
.log-entry.turn{{color:#b0bec5;font-size:10px;opacity:0.7}}
.log-entry.draw{{color:#b0bec5;opacity:0.6}}
.log-entry.misc{{color:#78909c}}

/* コンテキスト */
.ctx-box{{background:#0a1a0b;border:1px solid #1b5e20;border-radius:6px;padding:8px;font-size:11px;color:#81c784}}
.ctx-box h3{{font-size:10px;color:#66bb6a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}}
.ctx-name{{font-weight:bold;color:#ffd700}}

/* カード詳細パネル */
#card-detail{{background:#0a1a0b;border:1px solid #2e7d32;border-radius:8px;padding:10px;font-size:11px;flex-shrink:0;min-height:80px}}
#card-detail h3{{font-size:10px;color:#66bb6a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid #1b5e20;padding-bottom:4px}}
#card-detail-content{{color:#c8e6c9}}
.detail-card-name{{font-size:15px;font-weight:bold;color:#ffd700;margin-bottom:2px}}
.detail-card-en{{font-size:10px;color:#558b5a;margin-bottom:6px}}
.detail-hp{{font-size:12px;color:#a5d6a7;margin-bottom:4px}}
.detail-section{{margin-top:6px;padding-top:4px;border-top:1px solid #1b5e20}}
.detail-section-title{{font-size:10px;color:#66bb6a;font-weight:bold;margin-bottom:3px;text-transform:uppercase}}
.detail-attack{{margin-bottom:6px}}
.detail-attack-name{{font-weight:bold;color:#ef9a9a;font-size:11px}}
.detail-attack-cost{{display:flex;gap:2px;flex-wrap:wrap;margin:2px 0}}
.detail-attack-dmg{{color:#ffd700;font-weight:bold;font-size:12px}}
.detail-attack-text{{color:#b0bec5;font-size:10px;line-height:1.4;margin-top:2px}}
.detail-ability-name{{font-weight:bold;color:#a5d6a7;font-size:11px}}
.detail-ability-text{{color:#b0bec5;font-size:10px;line-height:1.4;margin-top:2px}}
.detail-meta{{color:#78909c;font-size:10px;margin-top:4px}}
.detail-empty{{color:#2e4030;font-style:italic}}

/* 弱点・耐性 */
.type-badge{{display:inline-block;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:bold;margin-left:4px}}

/* 選択中プレイヤー */
.deciding-player .active-card{{animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%{{box-shadow:0 0 6px #ffd70066}}50%{{box-shadow:0 0 20px #ffd700cc}}100%{{box-shadow:0 0 6px #ffd70066}}}}
</style>
</head>
<body>

<div id="header">
  <h1>⚡ Pokemon TCG Battle Replay</h1>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <span class="info-badge" id="turn-badge">Turn 0</span>
    <span class="info-badge" id="player-badge">P0 選択中</span>
    <span class="info-badge" id="result-badge">進行中</span>
  </div>
</div>

<div id="nav">
  <button onclick="goTo(0)" title="最初">⏮</button>
  <button id="btn-prev" onclick="prev()" title="前">◀</button>
  <input  id="step-slider" type="range" min="0" max="0" value="0" oninput="goTo(parseInt(this.value))">
  <span   id="step-label">0 / 0</span>
  <button id="btn-next" onclick="next()" title="次">▶</button>
  <button onclick="goTo(STATES.length-1)" title="最後">⏭</button>
  <button id="btn-play" onclick="togglePlay()">▶ 自動再生</button>
  <select id="speed-select" onchange="setSpeed(this.value)">
    <option value="1500">×0.7</option>
    <option value="1000">×1</option>
    <option value="500" selected>×2</option>
    <option value="250">×4</option>
    <option value="100">×10</option>
  </select>
  <input type="file" id="json-input" accept=".json" style="display:none" onchange="loadJson(event)">
  <button id="load-btn" onclick="document.getElementById('json-input').click()">📂 JSONを読み込む</button>
</div>

<div id="main">
  <div id="board">
    <!-- Opponent (top) -->
    <div class="player-side" id="opp-side">
      <div class="prize-zone">
        <div class="prize-label">サイド</div>
        <div class="prize-stack" id="opp-prizes"></div>
        <div class="prize-count" id="opp-prize-count">6</div>
      </div>
      <div class="field-zone">
        <div class="bench-row" id="opp-bench"></div>
        <div style="display:flex;justify-content:center">
          <div id="opp-active"></div>
        </div>
      </div>
      <div class="deck-zone">
        <div class="deck-pile deck-back"><div class="deck-pile-label">山</div><div class="deck-pile-count" id="opp-deck">0</div></div>
        <div class="deck-pile discard-back"><div class="deck-pile-label">トラッシュ</div><div class="deck-pile-count" id="opp-disc">0</div></div>
      </div>
    </div>

    <div class="deciding-indicator" id="deciding-indicator">── ── ──</div>

    <div class="board-divider">
      <div class="divider-line"></div>
      <div class="divider-vs">⚔ VS ⚔</div>
      <div class="divider-line"></div>
    </div>

    <!-- You (bottom) -->
    <div class="player-side" id="my-side">
      <div class="prize-zone">
        <div class="prize-label">サイド</div>
        <div class="prize-stack" id="my-prizes"></div>
        <div class="prize-count" id="my-prize-count">6</div>
      </div>
      <div class="field-zone">
        <div style="display:flex;justify-content:center">
          <div id="my-active"></div>
        </div>
        <div class="bench-row" id="my-bench"></div>
      </div>
      <div class="deck-zone">
        <div class="deck-pile deck-back"><div class="deck-pile-label">山</div><div class="deck-pile-count" id="my-deck">0</div></div>
        <div class="deck-pile discard-back"><div class="deck-pile-label">トラッシュ</div><div class="deck-pile-count" id="my-disc">0</div></div>
      </div>
    </div>
  </div>

  <!-- 右パネル -->
  <div id="log-panel">
    <div class="log-box">
      <h3>バトルログ</h3>
      <div id="log-entries"></div>
    </div>
    <div class="ctx-box" id="ctx-box">
      <h3>選択コンテキスト</h3>
      <div id="ctx-info">-</div>
    </div>
    <div id="card-detail">
      <h3>カード詳細</h3>
      <div id="card-detail-content"><div class="detail-empty">カードをクリックで詳細表示</div></div>
    </div>
  </div>
</div>

<div id="hand-zone">
  <h3>P0 の手札</h3>
  <div id="hand-cards"></div>
</div>

<script>
let STATES = {states_json};
const CARD_INFO = {card_info_json};
const ENERGY_COLOR = {energy_color};
const ENERGY_LABEL = {energy_label};
const ENERGY_NAME  = {energy_name};

const CTX_NAMES = {{
  0:"メインフェーズ", 1:"バトル場選択(セットアップ)", 2:"ベンチ選択(セットアップ)",
  3:"バトル場交代", 4:"バトル場へ", 5:"ベンチへ", 7:"手札へ",
  8:"トラッシュ選択", 14:"ダメカン配置", 25:"効果の対象",
  26:"エネルギートラッシュ", 35:"ワザ選択",
}};

const STAGE_JP = {{ basic:"たね", stage1:"1進化", stage2:"2進化" }};
const CARDTYPE_JP = {{ 0:"ポケモン",1:"グッズ",2:"ポケモンのどうぐ",3:"サポート",4:"スタジアム",5:"基本エネルギー",6:"特殊エネルギー" }};

let currentStep = 0;
let playing = false;
let playTimer = null;
let playSpeed = 500;

function energyDot(e, small) {{
  const color = ENERGY_COLOR[e] || "#888";
  const label = ENERGY_LABEL[e] || "?";
  const sz = small ? "12px" : "15px";
  const fs = small ? "6.5px" : "7.5px";
  return `<span class="energy-dot" style="background:${{color}};width:${{sz}};height:${{sz}};font-size:${{fs}}" title="${{ENERGY_NAME[e]||label}}">${{label}}</span>`;
}}

function pokemonCardHTML(pk, isActive) {{
  if (!pk) {{
    const cls = isActive ? "pokemon-card active-card empty-slot" : "pokemon-card empty-slot";
    return `<div class="${{cls}}"><div style="font-size:9px;color:#2e4030;margin:auto">空</div></div>`;
  }}
  const info = CARD_INFO[pk.id] || {{}};
  const jpName = info.jpName || info.name || `#${{pk.id}}`;
  const enName = info.name || "";
  const isEx = info.ex;
  const isTera = info.tera;

  const hp = pk.hp, maxHp = pk.maxHp || hp || 1;
  const ratio = maxHp > 0 ? hp / maxHp : 0;
  const hpClass = ratio > 0.5 ? "hp-high" : ratio > 0.25 ? "hp-mid" : "hp-low";
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));

  let statusHTML = "";
  if (pk.poisoned)  statusHTML += `<span class="status-badge status-psn">毒</span>`;
  if (pk.burned)    statusHTML += `<span class="status-badge status-brn">やけど</span>`;
  if (pk.asleep)    statusHTML += `<span class="status-badge status-slp">ねむり</span>`;
  if (pk.paralyzed) statusHTML += `<span class="status-badge status-par">まひ</span>`;
  if (pk.confused)  statusHTML += `<span class="status-badge status-cnf">こんらん</span>`;

  const energiesHTML = (pk.energies || []).map(e => energyDot(e, false)).join("");
  const toolsHTML = (pk.tools || []).map(tid => {{
    const ti = CARD_INFO[tid];
    return `<div class="tool-badge">${{ti ? ti.jpName : "#"+tid}}</div>`;
  }}).join("");

  const newBadge = pk.isNew ? `<span class="card-new-badge">NEW</span>` : "";
  const cls = isActive ? "pokemon-card active-card" : "pokemon-card";

  return `<div class="${{cls}}" onclick="showCardDetail(${{pk.id}})" data-card-id="${{pk.id}}">
    <div class="card-header">
      <div class="card-name-jp" title="${{jpName}}">${{jpName}}</div>
      ${{isEx ? `<span class="card-ex">ex</span>` : ""}}
    </div>
    <div class="card-name-en">${{enName}}</div>
    ${{newBadge}}
    ${{isTera ? `<span class="card-tera">☆テラ</span>` : ""}}
    <div class="hp-bar-container">
      <div class="hp-bar-fill ${{hpClass}}" style="width:${{pct}}%"></div>
    </div>
    <div class="hp-text">${{hp}}/${{maxHp}}</div>
    <div class="energies">${{energiesHTML}}</div>
    ${{statusHTML ? `<div class="status-badges">${{statusHTML}}</div>` : ""}}
    ${{toolsHTML}}
  </div>`;
}}

function prizeStackHTML(count) {{
  let h = "";
  for (let i = 0; i < Math.min(count, 6); i++) h += `<div class="prize-card">✦</div>`;
  return h;
}}

function handCardHTML(cid) {{
  const info = CARD_INFO[cid] || {{}};
  const ct = info.cardType ?? -1;
  const jpName = info.jpName || info.name || `#${{cid}}`;
  return `<span class="hand-card type-${{ct}}" onclick="showCardDetail(${{cid}})" title="${{info.name||cid}}">${{jpName}}</span>`;
}}

function showCardDetail(cid) {{
  const info = CARD_INFO[cid];
  if (!info) {{ document.getElementById("card-detail-content").innerHTML = `<div class="detail-empty">#${{cid}} データなし</div>`; return; }}

  const et = info.energyType;
  const etColor = ENERGY_COLOR[et] || "#888";
  const etName  = ENERGY_NAME[et]  || "";
  const stage = STAGE_JP[info.stage] || info.stage;
  let h = `<div class="detail-card-name">${{info.jpName}}</div>`;
  h += `<div class="detail-card-en">${{info.name}}</div>`;

  if (info.cardType === 0) {{
    // ポケモン
    h += `<div class="detail-hp">HP ${{info.hp}} | ${{stage}} | <span style="color:${{etColor}}">${{etName}}タイプ</span></div>`;
    if (info.evolvesFrom) h += `<div class="detail-meta">進化元: ${{info.evolvesFrom}}</div>`;

    // 弱点・耐性
    let metaStr = "";
    if (info.weakness >= 0) {{
      const wColor = ENERGY_COLOR[info.weakness] || "#888";
      metaStr += ` 弱点:<span style="color:${{wColor}}">${{ENERGY_NAME[info.weakness]||info.weakness}}</span>×2`;
    }}
    if (info.resistance >= 0) {{
      const rColor = ENERGY_COLOR[info.resistance] || "#888";
      metaStr += ` 耐性:<span style="color:${{rColor}}">${{ENERGY_NAME[info.resistance]||info.resistance}}</span>-30`;
    }}
    // にげるコスト
    const retreat = Array(info.retreatCost).fill(0).map(e => energyDot(0, true)).join("");
    metaStr += ` にげる:${{retreat||"0"}}`;
    if (metaStr) h += `<div class="detail-meta">${{metaStr}}</div>`;

    // アビリティ
    if (info.abilities && info.abilities.length > 0) {{
      h += `<div class="detail-section"><div class="detail-section-title">特性</div>`;
      for (const ab of info.abilities) {{
        h += `<div class="detail-attack"><div class="detail-ability-name">【${{ab.name}}】</div>`;
        if (ab.text) h += `<div class="detail-ability-text">${{ab.text}}</div>`;
        h += `</div>`;
      }}
      h += `</div>`;
    }}

    // ワザ
    if (info.attacks && info.attacks.length > 0) {{
      h += `<div class="detail-section"><div class="detail-section-title">ワザ</div>`;
      for (const atk of info.attacks) {{
        const cost = (atk.energies || []).map(e => energyDot(e, true)).join("");
        h += `<div class="detail-attack">`;
        h += `<div class="detail-attack-name">${{atk.name}} <span class="detail-attack-dmg">${{atk.damage > 0 ? atk.damage+"ダメージ" : ""}}</span></div>`;
        h += `<div class="detail-attack-cost">${{cost||"コストなし"}}</div>`;
        if (atk.text) h += `<div class="detail-attack-text">${{atk.text}}</div>`;
        h += `</div>`;
      }}
      h += `</div>`;
    }}
  }} else {{
    // トレーナーズ・エネルギー
    const ct = CARDTYPE_JP[info.cardType] || "カード";
    h += `<div class="detail-hp" style="color:#90caf9">${{ct}}</div>`;
    if (info.aceSpec) h += `<div class="detail-meta" style="color:#ffd700">★ ACE SPEC</div>`;
    if (info.abilities && info.abilities.length > 0) {{
      const ab = info.abilities[0];
      h += `<div class="detail-section"><div class="detail-ability-text">${{ab.text}}</div></div>`;
    }}
  }}

  document.getElementById("card-detail-content").innerHTML = h;
}}

function render(idx) {{
  if (idx < 0 || idx >= STATES.length) return;
  currentStep = idx;
  const s = STATES[idx];
  const p0 = s.players[0] || {{}};
  const p1 = s.players[1] || {{}};

  document.getElementById("turn-badge").textContent = `Turn ${{s.turn}}`;
  document.getElementById("player-badge").textContent = `P${{s.player}} 選択中`;
  const rb = document.getElementById("result-badge");
  if (s.result === -1) {{ rb.textContent = "進行中"; rb.className = "info-badge"; }}
  else {{ rb.textContent = `P${{s.result}} の勝利！`; rb.className = `info-badge win${{s.result}}`; }}

  document.getElementById("step-slider").value = idx;
  document.getElementById("step-label").textContent = `${{idx}} / ${{STATES.length-1}}`;
  document.getElementById("deciding-indicator").textContent =
    s.result === -1 ? `── P${{s.player}} が選択中 ──` : `── ゲーム終了 ──`;

  // オポーネント（P1）
  document.getElementById("opp-active").innerHTML = pokemonCardHTML(p1.active, true);
  const oppBench = [];
  for (let i = 4; i >= 0; i--) oppBench.push(pokemonCardHTML(p1.bench ? p1.bench[i]||null : null, false));
  document.getElementById("opp-bench").innerHTML = oppBench.join("");
  document.getElementById("opp-deck").textContent = p1.deckCount ?? "?";
  document.getElementById("opp-disc").textContent = p1.discardCount ?? "?";
  document.getElementById("opp-prize-count").textContent = p1.prizeCount ?? "?";
  document.getElementById("opp-prizes").innerHTML = prizeStackHTML(p1.prizeCount || 0);

  // 自分（P0）
  document.getElementById("my-active").innerHTML = pokemonCardHTML(p0.active, true);
  const myBench = [];
  for (let i = 0; i < 5; i++) myBench.push(pokemonCardHTML(p0.bench ? p0.bench[i]||null : null, false));
  document.getElementById("my-bench").innerHTML = myBench.join("");
  document.getElementById("my-deck").textContent = p0.deckCount ?? "?";
  document.getElementById("my-disc").textContent = p0.discardCount ?? "?";
  document.getElementById("my-prize-count").textContent = p0.prizeCount ?? "?";
  document.getElementById("my-prizes").innerHTML = prizeStackHTML(p0.prizeCount || 0);

  // 選択中ハイライト
  document.getElementById("opp-side").className = "player-side" + (s.player === 1 ? " deciding-player" : "");
  document.getElementById("my-side").className  = "player-side" + (s.player === 0 ? " deciding-player" : "");

  // 手札
  const handDiv = document.getElementById("hand-cards");
  if (p0.hand && p0.hand.length > 0) {{
    handDiv.innerHTML = p0.hand.map(handCardHTML).join("");
  }} else {{
    handDiv.innerHTML = `<span style="color:#2e4030;font-size:11px">${{p0.handCount||0}}枚（非公開）</span>`;
  }}

  // ログ
  const logDiv = document.getElementById("log-entries");
  if (s.logs && s.logs.length > 0) {{
    logDiv.innerHTML = s.logs.slice(-14).map(l =>
      `<div class="log-entry ${{l.cls}}">${{l.text}}</div>`
    ).join("");
    logDiv.scrollTop = logDiv.scrollHeight;
  }} else {{
    logDiv.innerHTML = `<div class="log-entry misc">（ログなし）</div>`;
  }}

  // コンテキスト
  const ctxDiv = document.getElementById("ctx-info");
  if (s.context !== null && s.context !== undefined) {{
    ctxDiv.innerHTML = `<span class="ctx-name">${{CTX_NAMES[s.context]||`context ${{s.context}}`}}</span>`;
    if (s.action) ctxDiv.innerHTML += `<br><span style="color:#81c784">選択: [${{s.action.join(", ")}}]</span>`;
  }} else {{
    ctxDiv.innerHTML = s.result !== -1 ? `<span style="color:#ffd700">ゲーム終了</span>` : `-`;
  }}

  document.getElementById("btn-prev").disabled = idx === 0;
  document.getElementById("btn-next").disabled = idx === STATES.length - 1;
}}

function goTo(idx) {{ render(Math.max(0, Math.min(STATES.length-1, idx))); }}
function prev() {{ goTo(currentStep - 1); }}
function next() {{ goTo(currentStep + 1); }}

function togglePlay() {{
  playing = !playing;
  document.getElementById("btn-play").textContent = playing ? "⏸ 一時停止" : "▶ 自動再生";
  if (playing) {{ if (currentStep >= STATES.length-1) goTo(0); autoPlay(); }}
  else clearTimeout(playTimer);
}}
function autoPlay() {{
  if (!playing || currentStep >= STATES.length-1) {{
    playing = false;
    document.getElementById("btn-play").textContent = "▶ 自動再生";
    return;
  }}
  next();
  playTimer = setTimeout(autoPlay, playSpeed);
}}
function setSpeed(v) {{ playSpeed = parseInt(v); }}

function loadJson(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {{
    try {{
      const data = JSON.parse(e.target.result);
      STATES = Array.isArray(data) ? data : (data.states || data);
      document.getElementById("step-slider").max = STATES.length - 1;
      goTo(0);
      alert(`読み込み完了: ${{STATES.length}}ステップ`);
    }} catch(err) {{
      alert("JSON の読み込みに失敗しました: " + err.message);
    }}
  }};
  reader.readAsText(file);
}}

document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft")  {{ prev(); e.preventDefault(); }}
  if (e.key === "ArrowRight") {{ next(); e.preventDefault(); }}
  if (e.key === " ")          {{ togglePlay(); e.preventDefault(); }}
}});

document.getElementById("step-slider").max = STATES.length - 1;
render(0);
</script>
</body>
</html>"""


# ── エントリポイント ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTCG バトルリプレイ HTML ジェネレータ")
    parser.add_argument("--from-json", default=None,
                        help="保存済み JSON バトルログファイル")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "battle_replay.html"),
                        help="出力 HTML ファイルパス")
    parser.add_argument("--copy-to-windows", action="store_true",
                        help="Windows Downloads にもコピー")
    args = parser.parse_args()

    print("カードデータ読み込み中...")
    cards   = all_card_data()
    attacks = all_attack()
    card_db    = {c.cardId: c for c in cards}
    attack_db  = {a.attackId: a for a in attacks}
    card_info  = build_card_info(card_db, attack_db)
    print(f"  カード {len(card_db)}枚  ワザ {len(attack_db)}件")

    if args.from_json:
        print(f"JSON 読み込み中: {args.from_json}")
        states = load_json_battle(args.from_json, card_db)
        print(f"  {len(states)} ステップ")
    else:
        print("対戦実行中...")
        states = run_battle_inline(card_db)
        final  = states[-1]
        winner = final.get("result", -1)
        print(f"  完了: {len(states)} ステップ  "
              f"Turn {final['turn']}  "
              f"結果: {'P'+str(winner)+'勝利' if winner != -1 else '不明'}")

    print(f"HTML 生成中: {args.out}")
    html = generate_html(states, card_info)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(args.out) // 1024
    print(f"  保存完了 ({size_kb} KB)")

    if args.copy_to_windows:
        win_path = "/mnt/c/Users/eriya/Downloads/battle_replay.html"
        try:
            shutil.copy2(args.out, win_path)
            print(f"  Windows にコピー: {win_path}")
        except Exception as e:
            print(f"  Windows コピー失敗: {e}")

    print("完了!")


if __name__ == "__main__":
    main()
