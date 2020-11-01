from typing import Any, Callable, Dict, Optional

import pyro
import torch

from .state import __MODULES, __STATE_DICT, Param
from ...session import get
from ...utility.tensor import checkpoint as _checkpoint


def get_module(
    name: str,
    module: Optional[Callable[[], torch.nn.Module]] = None,
    checkpoint: bool = False,
) -> Callable[..., Any]:
    r"""
    Retrieves :class:`~torch.nn.Module` by name or creates it if it doesn't
    exist.

    :param name: Module name
    :param module: Module to register if it doesn't already exist. The module
    should be "quoted" by encapsulating it in a `Callable` in order to lazify
    its creation.
    :param checkpoint: Flag indicating whether the module should be
    checkpointed

    :returns: The module
    :raises RuntimeError: If there is no module named `name` and `module` is
    `None`.
    """
    try:
        module_ = pyro.module(name, __MODULES[name])
    except KeyError as exc:
        if module is None:
            raise RuntimeError(f'Module "{name}" does not exist') from exc
        module_ = pyro.module(name, module(), update_module_params=True)
        if name in __STATE_DICT.modules:
            module_.load_state_dict(__STATE_DICT.modules[name])
        module_ = module_.to(get("default_device"))
        __MODULES[name] = module_
    module_ = module_.train(not get("eval"))
    if checkpoint:
        return lambda *args, **kwargs: _checkpoint(module_, *args, **kwargs)
    return module_


def get_param(
    name: str,
    default_value: Optional[Callable[[], torch.Tensor]] = None,
    lr_multiplier: float = 1.0,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Retrieves learnable :class:`~torch.Tensor` (non-module parameter) by
    name or creates it if it doesn't exist.

    :param name: Parameter name
    :param default_value: Default value if parameter doesn't exist. The value
    should be "quoted" by encapsulating it in a `Callable` in order to lazify
    its creation.
    :param lr_multiplier: Learning rate multiplier
    :param kwargs: Arguments passed to :func:`~pyro.sample`.

    :returns: The parameter
    :raises RuntimeError: If there is no parameter named `name` and
    `default_value` is `None`.
    """
    if name in pyro.get_param_store():
        return pyro.param(name)
    try:
        value = __STATE_DICT.params[name].data
    except KeyError as exc:
        if default_value is None:
            raise RuntimeError(f'Parameter "{name}" does not exist') from exc
        if callable(default_value):
            value = default_value()
        else:
            value = default_value
        __STATE_DICT.params[name] = Param(
            data=value.detach().cpu(),
            optim_args={"lr_multiplier": lr_multiplier},
        )
    return pyro.param(name, value.to(get("default_device")), **kwargs)


def get_param_optim_args(name: str) -> Dict[str, Any]:
    r"""
    :param name: Parameter name
    :returns: The optimizer arguments
    :raises KeyError: If there is no parameter named `name`
    """
    return __STATE_DICT.params[name].optim_args
