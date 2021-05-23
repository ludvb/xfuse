from copy import copy
from typing import Any, Dict, NamedTuple

import pyro
import torch

from ...session import get
from ...utility.tensor import to_device


class Param(NamedTuple):
    r"""Data structure for model parameters"""
    data: torch.Tensor
    optim_args: Dict[str, Any] = {}


class StateDict(NamedTuple):
    r"""Data structure for the states of modules and non-module parameters"""
    modules: Dict[str, Dict[str, torch.Tensor]]  # type: ignore
    params: Dict[str, Param]
    optimizer: Dict[str, Dict[str, torch.Tensor]]


__MODULES: Dict[str, torch.nn.Module] = {}
__STATE_DICT: StateDict = StateDict(modules={}, params={}, optimizer={})


def get_state_dict() -> StateDict:
    r"""Returns the state dicts of the modules in the module store"""
    state_dict = StateDict(
        modules=copy(__STATE_DICT.modules),
        params=copy(__STATE_DICT.params),
        optimizer=copy(__STATE_DICT.optimizer),
    )
    state_dict.modules.update(
        {
            name: to_device(module.state_dict(), torch.device("cpu"))
            for name, module in __MODULES.items()
        }
    )
    param_store = pyro.get_param_store()
    state_dict.params.update(
        {
            name: Param(
                data=param_store[name].detach().cpu(),
                optim_args=param.optim_args,
            )
            for name, param in __STATE_DICT.params.items()
            if name in param_store
        }
    )
    optimizer = get("optimizer")
    if optimizer is not None:
        state_dict.optimizer.update(
            to_device(optimizer.get_state(), torch.device("cpu"))
        )
    return state_dict


def load_state_dict(state_dict: StateDict) -> None:
    r"""Sets the default state dicts for the modules in the module store"""
    reset_state()
    __STATE_DICT.modules.update(state_dict.modules)
    __STATE_DICT.params.update(state_dict.params)
    __STATE_DICT.optimizer.update(state_dict.optimizer)
    optimizer = get("optimizer")
    if optimizer is not None:
        optimizer.set_state(__STATE_DICT.optimizer)


def reset_state() -> None:
    r"""Resets all state modules and parameters"""
    __MODULES.clear()
    __STATE_DICT.modules.clear()
    __STATE_DICT.params.clear()
    __STATE_DICT.optimizer.clear()
    pyro.clear_param_store()
    optimizer = get("optimizer")
    if optimizer is not None:
        optimizer.optim_objs.clear()
        optimizer.grad_clip.clear()
        # pylint: disable=protected-access
        optimizer._state_waiting_to_be_consumed.clear()
