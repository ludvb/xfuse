import torch

from .. import SessionItem, register_session_item


def _set_eval(eval_mode):
    torch.set_grad_enabled(not eval_mode)


register_session_item(
    "eval", SessionItem(setter=_set_eval, default=False, persistent=False)
)
