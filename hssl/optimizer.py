from inspect import signature

from typing import Optional

import torch as t

from .network import XFuse


def create_optimizer(
        model: XFuse,
        learning_rate: Optional[float] = None,
) -> t.optim.Optimizer:
    return t.optim.Adam(
        model.parameters(),
        lr=(
            learning_rate
            if learning_rate is not None else
            signature(t.optim.Adam).parameters['lr'].default
        ),
        amsgrad=True,
    )
