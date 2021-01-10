from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
    overload,
)

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..session import get


def checkpoint(function, *args, **kwargs):
    r"""
    Wrapper for :func:`torch.utils.checkpoint.checkpoint` that conditions
    checkpointing on the session `eval` state.
    """
    if get("eval"):
        return function(*args, **kwargs)
    return _checkpoint(function, *args, **kwargs)


class NoDevice(Exception):
    # pylint: disable=missing-class-docstring
    pass


def find_device(x: Any) -> torch.device:
    r"""
    Tries to find the :class:`torch.device` associated with the given object
    """

    if isinstance(x, torch.Tensor):
        return x.device

    if isinstance(x, list):
        for y in x:
            try:
                return find_device(y)
            except NoDevice:
                pass

    if isinstance(x, dict):
        for y in x.values():
            try:
                return find_device(y)
            except NoDevice:
                pass

    raise NoDevice(f"Failed to find a device associated with {x}")


def sparseonehot(labels: torch.Tensor, num_classes: Optional[int] = None):
    r"""One-hot encodes a label vectors into a sparse tensor"""
    if num_classes is None:
        num_classes = cast(int, labels.max().item()) + 1
    idx = torch.stack([torch.arange(labels.shape[0]).to(labels), labels])
    return torch.sparse.LongTensor(  # type: ignore
        idx,
        torch.ones(idx.shape[1]).to(idx),
        torch.Size([labels.shape[0], num_classes]),
    )


def isoftplus(x, /):
    r"""
    Inverse softplus.

    >>> ((isoftplus(torch.nn.functional.softplus(torch.linspace(-5, 5, 10)))
    ...     - torch.linspace(-5, 5, 10)) < 1e-5).all()
    tensor(True)
    """
    return np.log(np.exp(x) - 1)


@overload
def to_device(
    x: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: List[Any], device: Optional[torch.device] = None
) -> List[Any]:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: Dict[Any, Any], device: Optional[torch.device] = None
) -> Dict[Any, Any]:
    # pylint: disable=missing-function-docstring
    ...


def to_device(x, device=None):
    r"""
    Converts :class:`torch.Tensor` or a collection of :class:`torch.Tensor` to
    the given :class:`torch.device`
    """
    if device is None:
        device = get("default_device")
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(y, device) for y in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x
