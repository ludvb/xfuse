from typing import Callable, Dict, Optional

import pyro
import torch

__MODULES: Dict[str, torch.nn.Module] = {}


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
        return pyro.module(name, __MODULES[name])
    except KeyError:
        if module is not None:
            registered_module = pyro.module(
                name, module(), update_module_params=True
            )
            __MODULES[name] = registered_module
            return registered_module
        raise RuntimeError(f"Module '{name}' does not exist")


def clear_module_store() -> None:
    r"""Clears the module store"""
    __MODULES.clear()
