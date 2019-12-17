from typing import Callable, Dict, Optional, OrderedDict

import pyro
import torch

from ..session import get

__MODULES: Dict[str, torch.nn.Module] = {}
__DEFAULT_STATE_DICT: Dict[str, OrderedDict[str, torch.Tensor]] = {}


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
        module_ = pyro.module(name, module(), update_module_params=True)
        if name in __DEFAULT_STATE_DICT:
            module_.load_state_dict(__DEFAULT_STATE_DICT[name])
        __MODULES[name] = module_
    return module_


def get_state_dict() -> Dict[str, OrderedDict[str, torch.Tensor]]:
    r"""Returns the state dicts of the modules in the module store"""
    state_dicts = __DEFAULT_STATE_DICT.copy()
    state_dicts.update(
        {name: module.state_dict() for name, module in __MODULES.items()}
    )
    return state_dicts


def load_state_dict(
    state_dict: Dict[str, OrderedDict[str, torch.Tensor]]
) -> None:
    r"""Sets the default state dicts for the modules in the module store"""
    __MODULES.clear()
    __DEFAULT_STATE_DICT.clear()
    __DEFAULT_STATE_DICT.update(state_dict)
