"""
PTCG AI Battle Challenge - MCTS エージェント

cabt Engine の search_begin / search_step / search_end APIを使って
モンテカルロ木探索を行う。

NOTE: cgライブラリが提供するSearch APIの実際のシグネチャは
      cg/api.py を確認して調整すること。
"""

import math
import random
from collections import defaultdict
from typing import Optional


class MCTSNode:
    def __init__(self, parent=None, action: Optional[list[int]] = None):
        self.parent = parent
        self.action = action  # このノードに至るアクション
        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0  # 累積報酬
        self.untried_actions: Optional[list] = None  # 未展開の行動

    @property
    def q_value(self) -> float:
        return self.value / max(self.visits, 1)

    def ucb(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return self.q_value + c * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb(c))


class MCTSAgent:
    """
    cabt Engine のSearch APIを使ったMCTSエージェント。

    各ターンで search_begin → search_step × n_simulations → search_end
    を繰り返し、最も訪問回数が多い行動を選択する。
    """

    def __init__(self, n_simulations: int = 100, c: float = 1.414, depth: int = 10):
        self.n_simulations = n_simulations
        self.c = c
        self.depth = depth

    def __call__(self, obs_dict: dict) -> list[int]:
        """メインエントリーポイント"""
        select = obs_dict["select"]
        options = select["option"]
        max_count = select["maxCount"]

        if not options:
            return []

        if len(options) == 1:
            return [0]

        # cabt のSearch APIが利用可能な場合はMCTS
        try:
            import cg
            return self._mcts_with_cg(obs_dict, cg, options, max_count)
        except (ImportError, AttributeError):
            # フォールバック: ヒューリスティクス
            return self._heuristic_fallback(options, max_count)

    def _mcts_with_cg(self, obs_dict: dict, cg_module, options: list, max_count: int) -> list[int]:
        """cabt Search APIを使ったMCTS実装"""
        root = MCTSNode()
        root.untried_actions = list(range(len(options)))

        try:
            search_id = cg_module.search_begin(obs_dict)
        except Exception:
            return self._heuristic_fallback(options, max_count)

        for _ in range(self.n_simulations):
            node = root
            current_search_id = search_id

            # Selection
            while node.untried_actions is not None and len(node.untried_actions) == 0 and node.children:
                node = node.best_child(self.c)

            # Expansion
            if node.untried_actions and len(node.untried_actions) > 0:
                action_idx = random.choice(node.untried_actions)
                node.untried_actions.remove(action_idx)
                try:
                    child_obs, done, result = cg_module.search_step(current_search_id, [action_idx])
                    child = MCTSNode(parent=node, action=[action_idx])
                    child.untried_actions = list(range(len(child_obs["select"]["option"]))) if not done else []
                    node.children.append(child)
                    node = child
                    current_search_id = child_obs.get("search_id", current_search_id)
                    if done:
                        reward = self._terminal_reward(result)
                        self._backpropagate(node, reward)
                        continue
                except Exception:
                    break

            # Simulation（ランダムロールアウト）
            reward = self._rollout(current_search_id, cg_module)
            self._backpropagate(node, reward)

        try:
            cg_module.search_end(search_id)
        except Exception:
            pass

        if not root.children:
            return self._heuristic_fallback(options, max_count)

        best = max(root.children, key=lambda n: n.visits)
        return best.action or [0]

    def _rollout(self, search_id, cg_module, max_depth: int = 20) -> float:
        """ランダムポリシーでシミュレーション"""
        current_id = search_id
        for _ in range(max_depth):
            try:
                obs, done, result = cg_module.search_step(current_id, None)
                if done:
                    return self._terminal_reward(result)
                n_opt = len(obs["select"]["option"])
                if n_opt == 0:
                    break
                action = [random.randint(0, n_opt - 1)]
                obs, done, result = cg_module.search_step(current_id, action)
                if done:
                    return self._terminal_reward(result)
                current_id = obs.get("search_id", current_id)
            except Exception:
                break
        return 0.0

    def _terminal_reward(self, result) -> float:
        """終局状態から報酬を計算"""
        if isinstance(result, dict):
            winner = result.get("winner", result.get("result", None))
            if winner == 0:
                return 1.0
            elif winner == 1:
                return -1.0
        return 0.0

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _heuristic_fallback(self, options: list, max_count: int) -> list[int]:
        """MCTSが使えない時のフォールバック"""
        type_priority = {
            "attack": 5,
            "evolve": 4,
            "attach_energy": 3,
            "play_trainer": 2,
            "play_basic": 1,
            "retreat": 0,
            "pass": -1,
        }
        scored = []
        for i, opt in enumerate(options):
            t = opt.get("type", "pass") if isinstance(opt, dict) else "pass"
            score = type_priority.get(t, 0) + random.uniform(0, 0.1)
            scored.append((score, i))
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:max_count]]
