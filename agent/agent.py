"""
PTCG AI Battle Challenge - Agent
ランダムエージェント（ベースライン）

cabt Engine API:
  obs_dict = {
    "logs": [...],
    "current": {
      "players": [PlayerState, PlayerState],
      "stadium": Card | None,
      "turn": int,
      ...
    } | None,
    "select": {
      "option": [...],
      "maxCount": int
    }
  }

Agent returns: list[int]  (selected option indices)
"""

import random


def agent(obs_dict: dict) -> list[int]:
    """ランダムにlegal actionsを選択するベースライン"""
    select = obs_dict["select"]
    n_options = len(select["option"])
    max_count = select["maxCount"]

    if n_options == 0:
        return []

    k = min(max_count, n_options)
    return random.sample(range(n_options), k)
