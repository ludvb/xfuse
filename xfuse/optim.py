from functools import wraps
from inspect import signature

import pyro

from .logging import DEBUG, log
from .utility.state.getters import get_param_optim_args
from .session import get


__all__ = []


def _make_wrapped_constructor(constructor):
    @wraps(constructor)
    def _constructor(default_optim_args, *args, **kwargs):
        def _optim_args(_module_name, param_name):
            optim_args = default_optim_args.copy()
            try:
                param_optim_args = get_param_optim_args(param_name)
            except KeyError:
                param_optim_args = {}
            if "lr" not in optim_args:
                optim_args["lr"] = get("learning_rate")
            for k, value in param_optim_args.items():
                if k == "lr_multiplier":
                    log(
                        DEBUG,
                        f"Adjusting learning rate to {optim_args['lr']*value=}"
                        ' for parameter "%s"',
                        param_name,
                    )
                    optim_args["lr"] *= value
                else:
                    raise RuntimeError(f'Unknown optim arg "{k}"')
            return optim_args

        return constructor(_optim_args, *args, **kwargs)

    return _constructor


for __name, __constructor in [
    (k, v)
    for k, v in pyro.optim.__dict__.items()
    if callable(v) and "optim_args" in signature(v).parameters.keys()
]:
    locals()[__name] = _make_wrapped_constructor(__constructor)
    __all__.append(__name)
