from copy import deepcopy
from typing import Any, Callable, NoReturn, Union, cast

import numpy as np
import pyro as p
from pyro.poutine.messenger import Messenger

from ....handlers import Noop
from ....logging import INFO, WARNING, log
from ....session.session import Session, get, require
from ....utility import to_device
from ... import XFuse
from ...utility import compare
from . import ST


def purge_metagenes(xfuse: XFuse, num_samples: int = 1) -> None:
    r"""
    Purges superfluous metagenes and adds new ones based on the
    `metagene_expansion_strategy` of the current :class:`Session`
    """

    log(INFO, "evaluating metagenes")

    def _xfuse_without(n):
        reduced_xfuse = deepcopy(xfuse)
        reduced_xfuse.get_experiment("ST").remove_metagene(
            n, remove_params=False
        )
        return reduced_xfuse

    with Session(log_level=WARNING):
        st_experiment = cast(ST, xfuse.get_experiment("ST"))
        metagenes = st_experiment.metagenes

        if len(metagenes) == 1:
            contrib = list(metagenes)
            noncontrib = []
        else:
            reduced_models, ns = zip(
                *[(_xfuse_without(n).model, n) for n in metagenes]
            )

            def _eval_on(x):
                def _sample_once():
                    guide = p.poutine.trace(xfuse.guide).get_trace(x)
                    full, *reduced = compare(
                        x, guide, xfuse.model, *reduced_models
                    )
                    return [x - full for x in reduced]

                res = [_sample_once() for _ in range(num_samples)]
                return np.mean(res, 0)

            dataloader = require("dataloader")
            scores = np.mean([_eval_on(to_device(x)) for x in dataloader], 0)

            noncontrib = [
                n for res, n in reversed(sorted(zip(scores, ns))) if res >= 0
            ]
            contrib = [n for n in ns if n not in noncontrib]

    log(
        INFO,
        "contributing metagenes: %s",
        ", ".join(contrib) if contrib != [] else "-",
    )
    log(
        INFO,
        "non-contributing metagenes: %s",
        ", ".join(noncontrib) if noncontrib != [] else "-",
    )

    expand_metagenes = get("metagene_expansion_strategy")
    expand_metagenes(xfuse.get_experiment("ST"), contrib, noncontrib)


class MetagenePurger(Messenger):
    r"""
    Runs :func:`purge_metagenes` at a fixed interval to purge superfluous
    metagenes or add new ones
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
        self, period: Union[int, Callable[[int], bool]] = 1, **kwargs: Any
    ):
        super().__init__()
        self._predicate = (
            period
            if callable(period)
            else lambda epoch: epoch % cast(int, period) == 0
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
                purge_metagenes(self._model, **self._kwargs)
