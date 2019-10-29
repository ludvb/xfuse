from copy import deepcopy
from typing import Any, Callable, Iterable, NoReturn, Union, cast

import numpy as np
import pyro as p
from pyro.poutine.messenger import Messenger

from ....handlers import Noop
from ....logging import INFO, WARNING, log
from ....session.session import Session, get
from ....utility import to_device
from ... import XFuse
from ...utility import compare
from . import ST


def purge_factors(xfuse: XFuse, data: Iterable[Any], **kwargs: Any) -> None:
    r"""
    Purges superfluous factors and adds new ones based on the
    `factor_expansion_strategy` of the current :class:`Session`
    """

    log(INFO, "evaluating factors")

    def _xfuse_without(n):
        reduced_xfuse = deepcopy(xfuse)
        reduced_xfuse.get_experiment("ST").remove_factor(
            n, remove_params=False
        )
        return reduced_xfuse

    with Session(log_level=WARNING):
        st_experiment = cast(ST, xfuse.get_experiment("ST"))
        factors = st_experiment.factors
        reduced_models, ns = zip(
            *[(_xfuse_without(n).model, n) for n in factors]
        )

        def _eval_on(x):
            guide = p.poutine.trace(xfuse.guide).get_trace(x)

            full, *reduced = compare(
                x, guide, xfuse.model, *reduced_models, **kwargs
            )
            scores = [x - full for x in reduced]

            return scores

        scores = np.stack([_eval_on(to_device(x)) for x in data]).mean(0)

    noncontrib = [
        n for res, n in reversed(sorted(zip(scores, ns))) if res >= 0
    ]
    contrib = [n for n in ns if n not in noncontrib]

    log(
        INFO,
        "contributing factors: %s",
        ", ".join(contrib) if contrib != [] else "-",
    )
    log(
        INFO,
        "non-contributing factors: %s",
        ", ".join(noncontrib) if noncontrib != [] else "-",
    )

    expand_factors = get("factor_expansion_strategy")
    expand_factors(xfuse.get_experiment("ST"), contrib, noncontrib)


class FactorPurger(Messenger):
    r"""
    Runs :func:`purge_factors` at a fixed interval to purge superfluous factors
    or add new ones
    """

    _model: XFuse

    def __new__(cls, *_args, **_kwargs):
        try:
            xfuse = get("model")
            _ = xfuse.get_experiment("ST")
        except (AttributeError, KeyError):
            log(
                WARNING,
                "session model does not have an ST experiment."
                f" {cls.__name__} will be disabled.",
            )
            return Noop()
        instance = super().__new__(cls)
        instance._model = xfuse
        return instance

    def __init__(
        self,
        data: Iterable[Any],
        frequency: Union[int, Callable[[int], bool]] = 1,
        **kwargs: Any,
    ):
        super().__init__()
        self._data = data
        self._predicate = (
            frequency
            if callable(frequency)
            else lambda epoch: epoch % cast(int, frequency) == 0
        )
        self._kwargs = kwargs

    def _handle(self, **_msg) -> NoReturn:
        # pylint: disable=no-self-use
        raise RuntimeError("unreachable code path")

    def _select_msg(self, **_msg) -> bool:
        # pylint: disable=no-self-use
        return False

    def _pyro_post_epoch(self, msg) -> None:
        if self._predicate(msg["kwargs"]["epoch"]):
            with Session(pyro_stack=[]):
                purge_factors(self._model, self._data, **self._kwargs)
