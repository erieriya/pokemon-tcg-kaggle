#!/usr/bin/env python3
"""
提出前ローカル検証スクリプト

Kaggle が agent() を呼ぶのとまったく同じ流れでテストする。
  1. submission/ ディレクトリを対象に main.py を import
  2. cg.game で対戦を回し、全ステップで agent() を呼ぶ
  3. クラッシュ・型エラー・不正な返り値を検出してレポート

使い方:
    python tools/test_agent.py           # submission/ をテスト (submit.sh 後に実行)
    python tools/test_agent.py --agent agent/dragapult_agent.py  # 直接ファイル指定
    python tools/test_agent.py --games 5                          # 5 試合ループ
"""

import sys, os, argparse, importlib.util, traceback, copy, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_BASE  = os.path.join(BASE_DIR, "data", "sample_submission")

sys.path.insert(0, CG_BASE)

from cg.game import battle_start, battle_select, battle_finish
from cg.api  import to_observation_class


# ── エージェント動的ロード ────────────────────────────────────────

def load_agent(agent_path: str):
    """指定ファイルを Kaggle と同じ方法でロード → agent 関数を返す"""
    agent_dir = os.path.dirname(os.path.abspath(agent_path))
    # cg/ を submission ディレクトリから解決できるよう追加
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    spec = importlib.util.spec_from_file_location("main", agent_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "agent"):
        raise RuntimeError(f"agent() 関数が見つかりません: {agent_path}")
    return mod.agent


# ── 1 試合テスト ─────────────────────────────────────────────────

def _call_agent(agent_fn, obs_dict: dict) -> list:
    """agent() を正しいディレクトリで呼び出す"""
    prev_cwd = os.getcwd()
    agent_dir = os.path.dirname(os.path.abspath(agent_fn.__code__.co_filename))
    try:
        os.chdir(agent_dir)
        return agent_fn(obs_dict)
    finally:
        os.chdir(prev_cwd)


def run_one_game(agent_fn, game_idx: int, max_steps: int = 2000) -> dict:
    """
    1 試合を完走し、結果を返す。
    返り値: {"ok": bool, "steps": int, "turns": int, "result": int, "errors": [str]}
    """
    errors = []
    step   = 0

    try:
        obs_dict, _ = battle_start(
            _read_deck(agent_fn),
            _read_deck(agent_fn),
        )
    except Exception as e:
        return {"ok": False, "steps": 0, "turns": 0, "result": -1,
                "errors": [f"battle_start 失敗: {e}"]}

    while step < max_steps:
        state  = obs_dict.get("current") or {}
        result = state.get("result", -1)
        sel    = obs_dict.get("select")

        if result != -1 or sel is None:
            break

        # ── agent() 呼び出し ──────────────────────────────────
        try:
            action = _call_agent(agent_fn, obs_dict)
        except Exception as e:
            errors.append(
                f"Step {step}: agent() 例外\n"
                + traceback.format_exc()
            )
            battle_finish()
            return {"ok": False, "steps": step, "turns": state.get("turn", 0),
                    "result": result, "errors": errors}

        # ── 返り値の検証 ──────────────────────────────────────
        n_opts = len((sel.get("option") or []))
        max_c  = sel.get("maxCount", 1)
        min_c  = sel.get("minCount", 1)
        ctx    = sel.get("context", -1)

        if not isinstance(action, (list, tuple)):
            errors.append(f"Step {step}: agent() が list を返さなかった (型={type(action).__name__})")
            battle_finish()
            return {"ok": False, "steps": step, "turns": state.get("turn", 0),
                    "result": result, "errors": errors}

        for idx in action:
            if not isinstance(idx, int):
                errors.append(f"Step {step}: action に int 以外が含まれる ({idx!r})")
                battle_finish()
                return {"ok": False, "steps": step, "turns": state.get("turn", 0),
                        "result": result, "errors": errors}
            if n_opts > 0 and not (0 <= idx < n_opts):
                errors.append(
                    f"Step {step}: インデックス {idx} が範囲外 (options={n_opts}, ctx={ctx})"
                )
                # 範囲外でも続行可能か確認するため continue する

        # ── 次のステップ ──────────────────────────────────────
        try:
            obs_dict = battle_select(action)
        except Exception as e:
            errors.append(f"Step {step}: battle_select 失敗: {e}")
            battle_finish()
            return {"ok": False, "steps": step, "turns": state.get("turn", 0),
                    "result": result, "errors": errors}

        step += 1

    try:
        battle_finish()
    except Exception:
        pass

    final_result = (obs_dict.get("current") or {}).get("result", -1)
    return {
        "ok":     len(errors) == 0,
        "steps":  step,
        "turns":  (obs_dict.get("current") or {}).get("turn", 0),
        "result": final_result,
        "errors": errors,
    }


def _read_deck(agent_fn) -> list:
    """agent() に obs_dict.select=None を渡してデッキを取得（agent ディレクトリで実行）"""
    prev_cwd = os.getcwd()
    agent_dir = os.path.dirname(os.path.abspath(agent_fn.__code__.co_filename))
    try:
        os.chdir(agent_dir)
        deck = agent_fn({"select": None, "current": None, "logs": []})
    finally:
        os.chdir(prev_cwd)
    # battle_start は chdir 後の deck を使う（グローバル状態なし）
    if not isinstance(deck, (list, tuple)) or len(deck) != 60:
        raise RuntimeError(f"デッキが 60 枚ではありません (got {len(deck) if hasattr(deck,'__len__') else type(deck)})")
    return list(deck)


# ── メイン ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="エージェント提出前ローカル検証")
    parser.add_argument(
        "--agent",
        default=os.path.join(BASE_DIR, "submission", "main.py"),
        help="テスト対象のエージェントファイル (デフォルト: submission/main.py)",
    )
    parser.add_argument("--games", type=int, default=3, help="試合数 (デフォルト: 3)")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    agent_path = os.path.abspath(args.agent)
    if not os.path.exists(agent_path):
        # submission/ がなければ agent/ を試す
        fallback = os.path.join(BASE_DIR, "agent", "dragapult_agent.py")
        if os.path.exists(fallback):
            print(f"[WARN] {agent_path} が見つからないため {fallback} を使用")
            agent_path = fallback
        else:
            print(f"[ERROR] エージェントファイルが見つかりません: {agent_path}")
            sys.exit(1)

    print(f"対象: {agent_path}")
    print("=" * 60)

    # ── import テスト ────────────────────────────────────────────
    print("[1/3] import テスト...")
    try:
        agent_fn = load_agent(agent_path)
        print("      OK - agent() 関数を確認")
    except Exception as e:
        print(f"      FAIL - import 失敗:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── デッキ取得テスト ─────────────────────────────────────────
    print("[2/3] デッキ取得テスト (select=None 呼び出し)...")
    try:
        deck = _read_deck(agent_fn)
        from collections import Counter
        cnt = Counter(deck)
        ace_specs = [cid for cid, n in cnt.items() if n >= 1 and cid == 1080]  # Unfair Stamp
        ace_count = sum(1 for cid in [1080] if cnt.get(cid, 0) > 0)
        print(f"      OK - {len(deck)} 枚  ユニーク {len(cnt)} 種")
        # ACE SPEC チェック (1枚まで)
        ace_ids = [1080]  # 既知の ACE SPEC ID
        total_ace = sum(cnt.get(a, 0) for a in ace_ids)
        if total_ace > 1:
            print(f"      WARN: ACE SPEC が {total_ace} 枚含まれています (最大 1 枚)")
    except Exception as e:
        print(f"      FAIL:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── 対戦テスト ───────────────────────────────────────────────
    print(f"[3/3] 対戦テスト ({args.games} 試合)...")
    all_ok   = True
    t_start  = time.time()

    for i in range(args.games):
        res = run_one_game(agent_fn, game_idx=i, max_steps=args.max_steps)
        status = "OK  " if res["ok"] else "FAIL"
        winner = f"P{res['result']} 勝利" if res["result"] != -1 else "不明"
        print(f"      [{status}] Game {i+1}: {res['steps']} steps, Turn {res['turns']}, {winner}")
        if res["errors"]:
            all_ok = False
            for err in res["errors"]:
                print(f"             ⚠ {err}")

    elapsed = time.time() - t_start
    print("=" * 60)
    if all_ok:
        print(f"✅ 全テスト PASS ({elapsed:.1f}s) — 提出可能です")
    else:
        print(f"❌ テスト FAIL ({elapsed:.1f}s) — 提出前に修正してください")
        sys.exit(1)


if __name__ == "__main__":
    main()
