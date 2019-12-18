from copy import copy
from typing import Any, Callable, Dict, NamedTuple, Optional, OrderedDict

import pyro
import torch

from ..session import get


class StateDict(NamedTuple):
    r"""Data structure for the states of modules and non-module parameters"""
    modules: Dict[str, OrderedDict[str, torch.Tensor]]
    params: Dict[str, torch.Tensor]


__MODULES: Dict[str, torch.nn.Module] = {}
__STATE_DICT: StateDict = StateDict(
    modules={}, params={},
)


def get_module(
    name: str, module: Optional[Callable[[], torch.nn.Module]] = None
) -> torch.nn.Module:
    r"""
    Retrieves :class:`~torch.nn.Module` by name or creates it if it doesn't
    exist.

    :param name: Module name
    :param module: Module to register if it doesn't already exist. The module
    should be "quoted" by encapsulating it in a `Callable` in order to lazify
    its creation.

    :returns: The module
    :raises RuntimeError: If there is no module named `name` and `module` is
    `None`.
    """
    try:
        module_ = pyro.module(name, __MODULES[name])
    except KeyError:
        if module is None:
            raise RuntimeError(f'Module "{name}" does not exist')
        module_ = pyro.module(name, module())
        if name in __STATE_DICT.modules:
            module_.load_state_dict(__STATE_DICT.modules[name])
        __MODULES[name] = module_
    return module_.train(not get("eval"))


def get_param(
    name: str,
    default_value: Optional[Callable[[], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Retrieves learnable :class:`~torch.Tensor` (non-module parameter) by
    name or creates it if it doesn't exist.

    :param name: Parameter name
    :param default_value: Default value if parameter doesn't exist. The value
    should be "quoted" by encapsulating it in a `Callable` in order to lazify
    its creation.
    :param kwargs: Arguments passed to :func:`~pyro.sample`.

    :returns: The parameter
    :raises RuntimeError: If there is no parameter named `name` and
    `default_value` is `None`.
    """
    try:
        param = pyro.param(name, __STATE_DICT.params[name], **kwargs)
    except KeyError:
        if default_value is None:
            raise RuntimeError(f'Parameter "{name}" does not exist')
        param = pyro.param(name, default_value(), **kwargs)
        __STATE_DICT.params[name] = param
    return param.requires_grad_(not get("eval"))


def get_state_dict() -> StateDict:
    r"""Returns the state dicts of the modules in the module store"""
    state_dicts = copy(__STATE_DICT)
    state_dicts.modules.update(
        {name: module.state_dict() for name, module in __MODULES.items()}
    )
    return state_dicts


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
