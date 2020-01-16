import torch as t

from .. import SessionItem, register_session_item
from ...utility.utility import to_device
from ...utility.state.state import StateDict, get_state_dict, load_state_dict


def _set_default_device(device):
    state_dict = get_state_dict()
    new_state_dict = StateDict(
        params={
            k: v.detach().to(device).requires_grad_()
            for k, v in state_dict.params.items()
        },
        modules=to_device(state_dict.modules, device=device),
        optimizer=to_device(state_dict.optimizer, device=device),
    )
    load_state_dict(new_state_dict)


register_session_item(
    "default_device",
    SessionItem(
        setter=_set_default_device,
        default=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
    ),
)
