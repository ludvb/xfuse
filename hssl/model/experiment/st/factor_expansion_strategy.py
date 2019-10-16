from abc import ABC, abstractmethod

from functools import reduce

from inspect import isabstract, isclass

from typing import Callable, List, Optional, Set

import torch as t

from . import ST, FactorDefault
from ....logging import DEBUG, log
from ....session import register_session_item
from ....session import SessionItem


class ExpansionStrategy(ABC):
    @abstractmethod
    def __call__(
        self,
        experiment: ST,
        contributing_factors: List[str],
        noncontributing_factors: List[str],
    ) -> None:
        pass


class ExtraBaselines(ExpansionStrategy):
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
            default = t.stack(defaults).mean(0).detach()

        scale_biases = [
            experiment._get_factor_decoder(1, n)[-1].bias.squeeze()
            for n in experiment.factors.keys()
        ]
        scale = t.stack(scale_biases).min().item()

        for _ in range(self.extra_factors - len(noncontributing_factors)):
            experiment.add_factor(FactorDefault(scale, default))
        for n in noncontributing_factors[
            : len(noncontributing_factors) - self.extra_factors
        ]:
            experiment.remove_factor(n, remove_params=False)


class Node:
    @abstractmethod
    def get_nodes(self):
        pass


class Split(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

    def get_nodes(self):
        return [*self.a.get_nodes(), *self.b.get_nodes()]


class Leaf(Node):
    def __init__(self, name: str, contributing: bool = False):
        self.name = name
        self.contributing = contributing

    def get_nodes(self):
        return [self.name]


def _map_modify(root: Node, fn: Callable[[Leaf], None]):
    if isinstance(root, Leaf):
        fn(root)
        return
    if isinstance(root, Split):
        _map_modify(root.a, fn)
        _map_modify(root.b, fn)
        return
    raise NotImplementedError()


def _show(root: Node) -> str:
    if isinstance(root, Split):
        return f"({_show(root.a)}), ({_show(root.b)})"
    if isinstance(root, Leaf):
        return f"{root.name}: {root.contributing}"
    raise NotImplementedError()


class RetractAndSplit(ExpansionStrategy):
    def __init__(self):
        self._root_nodes: Set[Node] = set()

    def __call__(
        self,
        experiment: ST,
        contributing_factors: List[str],
        noncontributing_factors: List[str],
    ) -> None:
        contrib = set(contributing_factors)
        noncontrib = set(noncontributing_factors)

        def _set_contributing(x: Leaf):
            x.contributing = x.name in contrib

        def _drop_nonexistant_branches(root: Node) -> Optional[Node]:
            if isinstance(root, Split):
                a = _drop_nonexistant_branches(root.a)
                b = _drop_nonexistant_branches(root.b)
                if a and b:
                    return root
                if a and not b:
                    return a
                if b and not a:
                    return b
                return None
            if isinstance(root, Leaf):
                if root.name in set.union(contrib, noncontrib):
                    return root
                return None
            raise NotImplementedError()

        def _drop_noncontributing_branches(root: Node) -> Optional[Node]:
            if isinstance(root, Split):
                a = _drop_noncontributing_branches(root.a)
                b = _drop_noncontributing_branches(root.b)
                if (a and b) or (
                    isinstance(root.a, Leaf)
                    and isinstance(root.b, Leaf)
                    and (a or b)
                ):
                    return Split(a or root.a, b or root.b)
                if a and not b:
                    return a
                if b and not a:
                    return b
                if not a and not b:
                    return None
            if isinstance(root, Leaf):
                if root.contributing:
                    return root
                return None
            raise NotImplementedError()

        def _extend_contributing_branches(root: Node) -> Node:
            if isinstance(root, Split):
                if (not isinstance(root.a, Leaf) or root.a.contributing) and (
                    not isinstance(root.b, Leaf) or root.b.contributing
                ):
                    return Split(
                        _extend_contributing_branches(root.a),
                        _extend_contributing_branches(root.b),
                    )
                return root
            if isinstance(root, Leaf):
                if root.contributing:
                    return Split(
                        root, Leaf(experiment.split_factor(root.name))
                    )
                return root
            raise NotImplementedError()

        def _log_trees(title: str):
            log(DEBUG, "%s:", title)
            for tree in self._root_nodes:
                log(DEBUG, "  %s", _show(tree))

        self._root_nodes = set(
            [
                tree
                for tree in map(_drop_nonexistant_branches, self._root_nodes)
                if tree is not None
            ]
        )

        for tree in self._root_nodes:
            _map_modify(tree, _set_contributing)

        _log_trees("trees before retraction")

        self._root_nodes = set(
            [
                tree
                for tree in map(
                    _drop_noncontributing_branches, self._root_nodes
                )
                if tree is not None
            ]
        )

        _log_trees("trees after retraction / before splitting")

        forest: Set[str] = reduce(
            lambda a, x: set.union(a, x),
            (x.get_nodes() for x in self._root_nodes),
            set(),
        )
        for x in contrib:
            if x not in forest:
                log(DEBUG, "adding new root node: %s", x)
                self._root_nodes.add(Leaf(x, True))
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


_factor_expansion_strategy = SessionItem(
    setter=_setter, default=ExtraBaselines(extra_factors=1)
)

register_session_item("factor_expansion_strategy", _factor_expansion_strategy)


STRATEGIES = {
    x.__name__: x
    for x in locals().values()
    if isclass(x)
    if issubclass(x, ExpansionStrategy)
    if not isabstract(x)
}
