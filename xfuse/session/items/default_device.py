import torch as t

from .. import SessionItem, register_session_item
from ...utility.utility import to_device_
from ...utility.state.state import StateDict, get_state_dict, load_state_dict


def _set_default_device(device):
    state_dict = get_state_dict()
    new_state_dict = StateDict(
        params=to_device_(state_dict.params, device=device),
        modules=to_device_(state_dict.modules, device=device),
    )
    load_state_dict(new_state_dict)


register_session_item(
    "default_device",
    SessionItem(
        setter=_set_default_device,
        default=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
    ),
)
