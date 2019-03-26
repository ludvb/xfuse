from inspect import signature

import itertools as it

from typing import Optional

import torch as t

from .network import Histonet, STD


def create_optimizer(
        histonet: Histonet,
        std: STD,
        learning_rate: Optional[float] = None,
) -> t.optim.Optimizer:
    return t.optim.Adam(
        it.chain(histonet.parameters(), std.parameters()),
        lr=(
            learning_rate
            if learning_rate is not None else
            signature(t.optim.Adam).parameters['lr'].default
        ),
        amsgrad=True,
    )
