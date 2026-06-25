"""
PTCGNet(agent/rl_agent.py)の方策・価値関数をガイドにした、PUCTツリーによる本格的なMCTS。

設計メモ(重要な実機検証で確定した事実):
- cg Engineのsearch_step(search_id, select)は、同じsearch_idを何度再利用しても
  (=同じ親ノードから何度でも)別の子ノード(新しいsearchId)を生成できる。
  親ノードは「消費」されない。これは複数の階層でも成り立つ(深さ2のノードのidを
  再利用して深さ2のさらに別の子を作ることもできる)。実機テストで確認済み。
  そのため、ノードをPythonオブジェクトとして保持しておけば、一度訪問した
  ノードは再度search_stepを呼ばずに(=エンジン呼び出しコスト無しで)何度でも
  辿れる、本格的なPUCTツリーのMCTSが実装できる。
- 子ノードは遅延展開する: 候補手(ネットワークの方策からサンプリング)は
  ノード展開時にprior(事前確率)付きで登録するが、実際にsearch_stepでその子の
  局面を確定させるのは、探索でその子が初めて選ばれた時(MCTSChild.search_state
  がNoneの間)だけ。
- 相手の不明情報(山札・サイド・手札・アクティブ)はプレースホルダーで埋める
  (kiyotah公式MCTSサンプルコードと同じ設計思想。精密な推測は不要)。
- search_stepが返すObservationはdataclassだが、dataclasses.asdict()で既存の
  encode_state/encode_actions(dict前提)にそのまま渡せることを確認済み。
- ノードの評価値はroot側(探索を呼び出した手番)の視点に統一する。評価(_expand)の
  時点でそのノードのyourIndexがrootと異なれば符号を反転し、以降のbackpropは
  単純な加算で済むようにする。
"""

import dataclasses
import math

import torch
import torch.nn.functional as F

from cg.api import Observation, search_begin, search_end, search_step, to_observation_class

from rl_agent import encode_actions, encode_state
from train_ppo import _sequential_sample

C_PUCT = 1.5


def _predict_unknowns(obs: Observation, my_deck_full: list[int]):
    """相手の不明情報をプレースホルダーで埋める。自分の山札/サイドは
    my_deck_full(60枚)の先頭から必要枚数を切り出すだけの簡易な近似。"""
    cur = obs.current
    my_index = cur.yourIndex
    me = cur.players[my_index]
    opp = cur.players[1 - my_index]
    placeholder = my_deck_full[0]
    your_deck = my_deck_full[: me.deckCount]
    your_prize = my_deck_full[: len(me.prize)]
    opp_active_list = opp.active or []
    opp_active_unknown = len(opp_active_list) == 1 and opp_active_list[0] is None
    return (
        your_deck,
        your_prize,
        [placeholder] * opp.deckCount,
        [placeholder] * len(opp.prize),
        [placeholder] * opp.handCount,
        [placeholder] if opp_active_unknown else [],
    )


class MCTSChild:
    __slots__ = ("action", "prior", "search_state", "node")

    def __init__(self, action, prior):
        self.action = action
        self.prior = prior
        self.search_state = None  # 初訪問時にsearch_stepで確定する
        self.node = None  # MCTSNode、初訪問時に作る


class MCTSNode:
    __slots__ = ("search_state", "parent", "children", "visit", "total_value", "terminal_value", "expanded")

    def __init__(self, search_state, parent=None):
        self.search_state = search_state
        self.parent = parent
        self.children: list[MCTSChild] = []
        self.visit = 0
        self.total_value = 0.0
        self.terminal_value = None
        self.expanded = False

    def value_mean(self) -> float:
        return self.total_value / self.visit if self.visit > 0 else 0.0

    def backprop(self, value: float):
        self.total_value += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def _terminal_value(result: int, root_my_index: int) -> float:
    if result == root_my_index:
        return 1.0
    if result in (0, 1):
        return -1.0
    return 0.0


def _sample_candidates(logits: torch.Tensor, n: int, k: int, num_candidates: int):
    max_distinct = 1
    try:
        max_distinct = math.comb(n, k)
    except ValueError:
        pass
    target = min(num_candidates, max_distinct)
    seen = set()
    candidates = []
    attempts = 0
    while len(candidates) < target and attempts < num_candidates * 8:
        attempts += 1
        action, log_prob = _sequential_sample(logits, k)
        key = tuple(sorted(action))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((action, math.exp(log_prob.item())))
    if not candidates:
        candidates = [(list(range(k)), 1.0)]
    total_prior = sum(p for _, p in candidates) or 1.0
    return [(action, p / total_prior) for action, p in candidates]


def _expand(node: MCTSNode, net, device: str, root_my_index: int, num_candidates: int) -> float:
    """未展開のノードを展開する。終局していればterminal_valueを設定する。
    戻り値はroot視点の評価値(backpropに使う)。
    """
    obs = node.search_state.observation
    cur = obs.current
    node.expanded = True
    if cur is not None and cur.result is not None and cur.result != -1:
        node.terminal_value = _terminal_value(cur.result, root_my_index)
        return node.terminal_value

    if cur is None or obs.select is None:
        node.terminal_value = 0.0
        return 0.0

    options = obs.select.option or []
    n = len(options)
    if n == 0:
        node.terminal_value = 0.0
        return 0.0
    k = max(obs.select.minCount, min(obs.select.maxCount, n))

    obs_as_dict = dataclasses.asdict(obs)
    state = encode_state(obs_as_dict, device)
    action_feats = encode_actions(options, obs_as_dict, device)
    with torch.no_grad():
        logits, value = net(state, action_feats)

    for action, prior in _sample_candidates(logits[0], n, k, num_candidates):
        node.children.append(MCTSChild(action, prior))

    sign = 1.0 if cur.yourIndex == root_my_index else -1.0
    return float(value.item()) * sign


def _select_child(node: MCTSNode) -> MCTSChild | None:
    """PUCT式で最も有望な子を選ぶ。"""
    best_child, best_score = None, float("-inf")
    sqrt_visit = math.sqrt(max(1, node.visit))
    for child in node.children:
        child_value = child.node.value_mean() if child.node is not None else 0.0
        child_visit = child.node.visit if child.node is not None else 0
        score = child_value + C_PUCT * child.prior * sqrt_visit / (1 + child_visit)
        if score > best_score:
            best_score, best_child = score, child
    return best_child


def _simulate(root: MCTSNode, net, device: str, root_my_index: int, num_candidates: int):
    """rootからPUCTで辿り、初めて訪れる子に到達したらsearch_stepで確定させて展開・評価し、backpropする。"""
    node = root
    while node.terminal_value is None and node.expanded and node.children:
        child = _select_child(node)
        if child is None:
            break
        if child.search_state is None:
            try:
                step_res = search_step(node.search_state.searchId, child.action)
            except Exception:
                step_res = None
            if step_res is None:
                # この子は展開できない(エンジンに拒否された等)。終局扱いの中立値にしておく。
                child.node = MCTSNode(node.search_state, parent=node)
                child.node.terminal_value = 0.0
                child.node.expanded = True
                child.node.backprop(0.0)
                return
            child.search_state = step_res
            child.node = MCTSNode(step_res, parent=node)
            value = _expand(child.node, net, device, root_my_index, num_candidates)
            child.node.backprop(value)
            return
        node = child.node
    if node.terminal_value is not None:
        node.backprop(node.terminal_value)
        return
    value = _expand(node, net, device, root_my_index, num_candidates)
    node.backprop(value)


def search_policy(
    obs_dict: dict,
    net,
    my_deck_full: list[int],
    device: str = "cpu",
    n_simulations: int = 64,
    num_candidates: int = 6,
):
    """root局面に対し、n_simulations回のPUCT探索シミュレーションを行い、
    (最多訪問の行動, [(action, 訪問回数で正規化した方策), ...]) を返す。
    探索できない/失敗した場合は (None, None) を返す。
    """
    obs = to_observation_class(obs_dict) if isinstance(obs_dict, dict) else obs_dict
    if obs.select is None or obs.current is None:
        return None, None
    if getattr(obs, "search_begin_input", None) is None:
        return None, None

    root_my_index = obs.current.yourIndex
    your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active = _predict_unknowns(obs, my_deck_full)

    try:
        res = search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
    except Exception:
        return None, None
    if res is None:
        return None, None

    root = MCTSNode(res, parent=None)
    try:
        value0 = _expand(root, net, device, root_my_index, num_candidates)
        root.backprop(value0)
        if root.terminal_value is not None or not root.children:
            return None, None
        for _ in range(n_simulations):
            _simulate(root, net, device, root_my_index, num_candidates)
    finally:
        try:
            search_end()
        except Exception:
            pass

    total_visits = sum((c.node.visit if c.node else 0) for c in root.children)
    if total_visits == 0:
        best = max(root.children, key=lambda c: c.prior)
        return best.action, [(c.action, c.prior) for c in root.children]
    policy_target = [(c.action, (c.node.visit if c.node else 0) / total_visits) for c in root.children]
    best = max(root.children, key=lambda c: (c.node.visit if c.node else 0))
    return best.action, policy_target
