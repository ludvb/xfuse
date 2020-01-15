from copy import copy
from typing import Dict, NamedTuple, OrderedDict

import torch


class StateDict(NamedTuple):
    r"""Data structure for the states of modules and non-module parameters"""
    modules: Dict[str, OrderedDict[str, torch.Tensor]]  # type: ignore
    params: Dict[str, torch.Tensor]


__MODULES: Dict[str, torch.nn.Module] = {}
__STATE_DICT: StateDict = StateDict(modules={}, params={})


def get_state_dict() -> StateDict:
    r"""Returns the state dicts of the modules in the module store"""
    state_dict = StateDict(
        modules=copy(__STATE_DICT.modules),
        params=copy(__STATE_DICT.params),
        optimizer=copy(__STATE_DICT.optimizer),
    )
    state_dict.modules.update(
        {name: module.state_dict() for name, module in __MODULES.items()}
    )
    return state_dict


def load_state_dict(state_dict: StateDict) -> None:
    r"""Sets the default state dicts for the modules in the module store"""
    reset_state()
    __STATE_DICT.modules.update(state_dict.modules)
    __STATE_DICT.params.update(state_dict.params)


def reset_state() -> None:
    r"""Resets all state modules and parameters"""
    __MODULES.clear()
    __STATE_DICT.modules.clear()
    __STATE_DICT.params.clear()
