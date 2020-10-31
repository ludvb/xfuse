from abc import ABC, abstractmethod
from functools import reduce
from inspect import isabstract, isclass
from typing import Callable, List, Optional, Set

import numpy as np

from ....logging import DEBUG, log
from ....session import SessionItem, register_session_item, get
from . import ST


class ExpansionStrategy(ABC):
    r"""Abstract base class for metagene expansion strategies"""

    @abstractmethod
    def __call__(
        self,
        experiment: ST,
        contributing_metagenes: List[str],
        noncontributing_metagenes: List[str],
    ) -> None:
        pass


class Extra(ExpansionStrategy):
    r"""
    An :class:`ExpansionStrategy` that keeps a number of "extra",
    non-contributing metagenes around. The number of extra metagenes can be
    fixed or linearly annealed.
    """

    def __init__(
        self,
        num_metagenes: int = 4,
        anneal_to: Optional[int] = 1,
        anneal_epochs: Optional[int] = 10000,
    ):
        self.__num_from = num_metagenes
        self.__num_to = anneal_to
        self.__anneal_epochs = anneal_epochs

    @property
    def num(self):
        r"""The current number of metagenes"""
        if self.__num_to is None or self.__anneal_epochs is None:
            return self.__num_from
        training_data = get("training_data")
        q = min(1.0, training_data.epoch / self.__anneal_epochs)
        return round((1 - q) * self.__num_from + q * self.__num_to)

    def __call__(
        self,
        experiment: ST,
        contributing_metagenes: List[str],
        noncontributing_metagenes: List[str],
    ) -> None:
        n_missing = self.num - len(noncontributing_metagenes)
        for _ in range(n_missing):
            experiment.add_metagene()
        for n, _ in zip(noncontributing_metagenes, range(-n_missing)):
            experiment.remove_metagene(n, remove_params=False)


class _Node:
    @abstractmethod
    def get_nodes(self):
        r"""Get all child nodes"""


class _Split(_Node):
    def __init__(self, a: _Node, b: _Node):
        self.a = a
        self.b = b

    def get_nodes(self):
        return [*self.a.get_nodes(), *self.b.get_nodes()]


class _Leaf(_Node):
    def __init__(self, name: str, contributing: bool = False):
        self.name = name
        self.contributing = contributing

    def get_nodes(self):
        return [self.name]


def _map_modify(root: _Node, fn: Callable[[_Leaf], None]):
    if isinstance(root, _Leaf):
        fn(root)
        return
    if isinstance(root, _Split):
        _map_modify(root.a, fn)
        _map_modify(root.b, fn)
        return
    raise NotImplementedError()


def _show(root: _Node) -> str:
    if isinstance(root, _Split):
        return f"({_show(root.a)}), ({_show(root.b)})"
    if isinstance(root, _Leaf):
        return f"{root.name}: {root.contributing}"
    raise NotImplementedError()


class DropAndSplit(ExpansionStrategy):
    r"""
    An :class:`ExpansionStrategy` that splits contributing metagenes and merges
    back previously split, non-contributing metagenes
    """

    def __init__(self, max_metagenes: int = 50):
        self._root_nodes: Set[_Node] = set()
        self._max_metagenes = max_metagenes

    def __call__(
        self,
        experiment: ST,
        contributing_metagenes: List[str],
        noncontributing_metagenes: List[str],
    ) -> None:
        contrib = set(contributing_metagenes)
        noncontrib = set(noncontributing_metagenes)

        def _set_contributing(x: _Leaf):
            x.contributing = x.name in contrib

        def _drop_nonexistant_branches(root: _Node) -> Optional[_Node]:
            if isinstance(root, _Split):
                a = _drop_nonexistant_branches(root.a)
                b = _drop_nonexistant_branches(root.b)
                if a and b:
                    return root
                if a and not b:
                    return a
                if b and not a:
                    return b
                return None
            if isinstance(root, _Leaf):
                if root.name in set.union(contrib, noncontrib):
                    return root
                return None
            raise NotImplementedError()

        def _drop_noncontributing_branches(root: _Node) -> _Node:
            if isinstance(root, _Split):
                if isinstance(root.b, _Leaf) and not root.b.contributing:
                    return _drop_noncontributing_branches(root.a)
                if isinstance(root.a, _Leaf) and not root.a.contributing:
                    return _drop_noncontributing_branches(root.b)
                return _Split(
                    *np.random.permutation(
                        [
                            _drop_noncontributing_branches(root.a),
                            _drop_noncontributing_branches(root.b),
                        ]
                    )
                )
            if isinstance(root, _Leaf):
                return root
            raise NotImplementedError()

        def _extend_contributing_branches(root: _Node) -> _Node:
            if isinstance(root, _Split):
                if isinstance(root.a, _Leaf) and isinstance(root.b, _Leaf):
                    if root.a.contributing and root.b.contributing:
                        return _Split(
                            _extend_contributing_branches(root.a),
                            _extend_contributing_branches(root.b),
                        )
                    return root
                return _Split(
                    *[
                        _extend_contributing_branches(node)
                        if not isinstance(node, _Leaf) or node.contributing
                        else node
                        for node in (root.a, root.b)
                    ]
                )
            if isinstance(root, _Leaf):
                if root.contributing and (
                    self._max_metagenes <= 0
                    or len(experiment.metagenes) < self._max_metagenes
                ):
                    return _Split(
                        root, _Leaf(experiment.split_metagene(root.name))
                    )
                return root
            raise NotImplementedError()

        def _log_trees(title: str):
            log(DEBUG, "%s:", title)
            for tree in self._root_nodes:
                log(DEBUG, "  %s", _show(tree))

        self._root_nodes = {
            tree
            for tree in map(_drop_nonexistant_branches, self._root_nodes)
            if tree is not None
        }

        for tree in self._root_nodes:
            _map_modify(tree, _set_contributing)

        _log_trees("Trees before retraction")

        self._root_nodes = set(
            map(_drop_noncontributing_branches, self._root_nodes)
        )

        # Remove non-contributing trees, keeping at least one
        noncontributing = [
            x
            for x in self._root_nodes
            if isinstance(x, _Leaf)
            if not x.contributing
        ]
        for tree in noncontributing[: len(self._root_nodes) - 1]:
            self._root_nodes.discard(tree)

        _log_trees("Trees after retraction / before splitting")

        forest: Set[str] = reduce(
            set.union, (x.get_nodes() for x in self._root_nodes), set()
        )
        for x in contrib:
            if x not in forest:
                log(DEBUG, "Adding new root node: %s", x)
                self._root_nodes.add(_Leaf(x, True))
        for x in noncontrib:
            if x not in forest:
                experiment.remove_metagene(x, remove_params=True)

        self._root_nodes = set(
            map(_extend_contributing_branches, self._root_nodes)
        )

        _log_trees("Trees after splitting")


register_session_item(
    "metagene_expansion_strategy",
    SessionItem(setter=lambda _: None, default=None, persistent=True),
)


STRATEGIES = {
    x.__name__: x
    for x in locals().values()
    if isclass(x)
    if issubclass(x, ExpansionStrategy)
    if not isabstract(x)
}
