"""
PTCG AI Battle Challenge - 学習済みRLエージェント 提出用エントリポイント

submit_rl.sh がこのファイルを main.py としてリネームし、
rl_agent.py / model_final.pt / deck.csv / cg/ と一緒にtar.gz化する。
"""

import os

from rl_agent import RLAgent

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(filename: str) -> str:
    """カレントディレクトリ優先、無ければ/kaggle_simulations/agent/を試す
    (dragapult_agent.pyのread_deck_csv()と同じ、提出環境向けの既存の流儀)"""
    if os.path.exists(filename):
        return filename
    return "/kaggle_simulations/agent/" + filename


def _read_deck() -> list[int]:
    with open(_resolve("deck.csv")) as f:
        lines = f.read().strip().split("\n")
    return [int(lines[i]) for i in range(60)]


_rl_agent: RLAgent | None = None


def _get_agent() -> RLAgent:
    # importではなくagent()呼び出し時に遅延ロードする。
    # Kaggle実行環境ではchdirが main.py のimportより後に行われるため、
    # import時点でmodel_final.pt(相対パス)を開こうとすると見つからない。
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent(model_path=_resolve("model_final.pt"), device="cpu")
    return _rl_agent


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return _read_deck()
    return _get_agent()(obs_dict)
