from copy import deepcopy
from typing import Any, Callable, NoReturn, Union, cast

import numpy as np
import pyro as p
from pyro.poutine.messenger import Messenger

from ....logging import INFO, WARNING, log
from ....session.session import Session, require
from ....utility.tensor import to_device
from ... import XFuse
from ...utility import compare
from . import ST
from .metagene_expansion_strategy import ExpansionStrategy


def purge_metagenes(num_samples: int = 1) -> None:
    r"""
    Purges superfluous metagenes and adds new ones based on the
    `metagene_expansion_strategy` of the current :class:`Session`
    """

    log(INFO, "Evaluating metagenes")

    xfuse: XFuse = require("model")
    metagene_expansion_strategy: ExpansionStrategy = require(
        "metagene_expansion_strategy"
    )

    def _xfuse_without(n):
        reduced_xfuse = deepcopy(xfuse)
        reduced_xfuse.get_experiment("ST").remove_metagene(
            n, remove_params=False
        )
        return reduced_xfuse

    with Session(log_level=WARNING, eval=True):
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
        "Contributing metagenes: %s",
        ", ".join(contrib) if contrib != [] else "-",
    )
    log(
        INFO,
        "Non-contributing metagenes: %s",
        ", ".join(noncontrib) if noncontrib != [] else "-",
    )

    metagene_expansion_strategy(st_experiment, contrib, noncontrib)


class MetagenePurger(Messenger):
    r"""
    Runs :func:`purge_metagenes` at a fixed interval to purge superfluous
    metagenes or add new ones
    """

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
        raise RuntimeError("Unreachable code path")

    def _select_msg(self, **_msg) -> bool:
        # pylint: disable=no-self-use
        return False

    def _pyro_post_epoch(self, msg) -> None:
        if self._predicate(msg["kwargs"]["epoch"]):
            with Session(messengers=[]):
                purge_metagenes(**self._kwargs)
