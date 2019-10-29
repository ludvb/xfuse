from abc import ABC, abstractmethod
from functools import reduce
from inspect import isabstract, isclass
from typing import Callable, List, Optional, Set

import torch

from ....logging import DEBUG, log
from ....session import SessionItem, register_session_item
from . import ST, FactorDefault


class ExpansionStrategy(ABC):
    r"""Abstract base class for factor expansion strategies"""

    @abstractmethod
    def __call__(
        self,
        experiment: ST,
        contributing_factors: List[str],
        noncontributing_factors: List[str],
    ) -> None:
        pass


class ExtraBaselines(ExpansionStrategy):
    r"""
    An :class:`ExpansionStrategy` that always keeps a fixed number of "extra",
    non-contributing factors around.
    """

    def __init__(self, extra_factors: int = 1):
        self.extra_factors = extra_factors

    def __call__(
        self,
        experiment: ST,
        contributing_factors: List[str],
        noncontributing_factors: List[str],
    ) -> None:
        defaults = [
            factor.profile
            for factor in experiment.factors.values()
            if factor.profile is not None
        ]
        if defaults == []:
            default = None
        else:
            default = torch.stack(defaults).mean(0).detach()

        scale_biases = [
            experiment._get_factor_decoder(1, n)[-1].bias.squeeze()
            for n in experiment.factors.keys()
        ]
        scale = torch.stack(scale_biases).min().item()

        for _ in range(self.extra_factors - len(noncontributing_factors)):
            experiment.add_factor(FactorDefault(scale, default))
        for n in noncontributing_factors[
            : len(noncontributing_factors) - self.extra_factors
        ]:
            experiment.remove_factor(n, remove_params=False)


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


class RetractAndSplit(ExpansionStrategy):
    r"""
    An :class:`ExpansionStrategy` that splits contributing factors and merges
    back previously split, non-contributing factors
    """

    def __init__(self):
        self._root_nodes: Set[_Node] = set()

    def __call__(
        self,
        experiment: ST,
        contributing_factors: List[str],
        noncontributing_factors: List[str],
    ) -> None:
        contrib = set(contributing_factors)
        noncontrib = set(noncontributing_factors)

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

        def _retract_noncontributing_branches(root: _Node) -> _Node:
            if isinstance(root, _Split):
                if isinstance(root.a, _Leaf) and isinstance(root.b, _Leaf):
                    if not (root.a.contributing or root.b.contributing):
                        return root.a
                    return _Split(root.a, root.b)
                a = _retract_noncontributing_branches(root.a)
                b = _retract_noncontributing_branches(root.b)
                return _Split(a, b)
            if isinstance(root, _Leaf):
                return root
            raise NotImplementedError()

        def _extend_contributing_branches(root: _Node) -> _Node:
            if isinstance(root, _Split):
                if (not isinstance(root.a, _Leaf) or root.a.contributing) and (
                    not isinstance(root.b, _Leaf) or root.b.contributing
                ):
                    return _Split(
                        _extend_contributing_branches(root.a),
                        _extend_contributing_branches(root.b),
                    )
                return root
            if isinstance(root, _Leaf):
                if root.contributing:
                    return _Split(
                        root, _Leaf(experiment.split_factor(root.name))
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

        _log_trees("trees before retraction")

        self._root_nodes = {
            tree
            for tree in map(
                _retract_noncontributing_branches, self._root_nodes
            )
            if not (isinstance(tree, _Leaf) and not tree.contributing)
        }

        _log_trees("trees after retraction / before splitting")

        forest: Set[str] = reduce(
            set.union, (x.get_nodes() for x in self._root_nodes), set()
        )
        for x in contrib:
            if x not in forest:
                log(DEBUG, "adding new root node: %s", x)
                self._root_nodes.add(_Leaf(x, True))
        for x in noncontrib:
            if x not in forest:
                experiment.remove_factor(x, remove_params=True)

        self._root_nodes = set(
            map(_extend_contributing_branches, self._root_nodes)
        )

        _log_trees("trees after splitting")


def _setter(x):
    if not isinstance(x, ExpansionStrategy):
        raise ValueError(f"{x} is not an expansion strategy")


register_session_item(
    "factor_expansion_strategy",
    SessionItem(setter=_setter, default=ExtraBaselines(extra_factors=1)),
)


STRATEGIES = {
    x.__name__: x
    for x in locals().values()
    if isclass(x)
    if issubclass(x, ExpansionStrategy)
    if not isabstract(x)
}
