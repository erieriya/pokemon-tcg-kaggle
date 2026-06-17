"""
PTCG AI Battle Challenge - メインエントリーポイント

提出ファイル:
  tar -czvf submission.tar.gz main.py deck.csv dragapult_agent.py cg/

AGENT_MODE を切り替えることで使うエージェントを選択できる:
  "dragapult" - Dragapult ex / Dusknoir デッキ専用ヒューリスティクス (デフォルト)
  "heuristic" - 汎用ルールベース
  "random"    - ランダム選択（デバッグ用）
  "rl"        - 学習済みRLエージェント（model_final.pt が必要）
"""

import os

AGENT_MODE = os.environ.get("PTCG_AGENT", "dragapult")
MODEL_PATH = os.environ.get("PTCG_MODEL_PATH", "model_final.pt")


def _load_agent():
    if AGENT_MODE == "rl" and os.path.exists(MODEL_PATH):
        from rl_agent import RLAgent
        return RLAgent(model_path=MODEL_PATH)
    elif AGENT_MODE == "heuristic":
        from heuristic_agent import agent
        return agent
    elif AGENT_MODE == "random":
        from agent import agent
        return agent
    else:  # "dragapult" (default)
        from dragapult_agent import agent
        return agent


_agent_fn = _load_agent()


def agent(obs_dict: dict) -> list[int]:
    return _agent_fn(obs_dict)
