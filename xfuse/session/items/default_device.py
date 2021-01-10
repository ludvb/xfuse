import torch

from .. import SessionItem, register_session_item
from ...logging import DEBUG, log
from ...utility.tensor import to_device
from ...utility.state.state import StateDict, get_state_dict, load_state_dict


__DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
__CURRENT_DEVICE = __DEFAULT_DEVICE


def _set_default_device(device):
    # pylint: disable=global-statement
    global __CURRENT_DEVICE
    if device != __CURRENT_DEVICE:
        log(DEBUG, "Setting default device to %s", str(device))
        state_dict = get_state_dict()
        new_state_dict = StateDict(
            params=state_dict.params,
            modules=to_device(state_dict.modules, device=device),
            optimizer=to_device(state_dict.optimizer, device=device),
        )
        load_state_dict(new_state_dict)
        __CURRENT_DEVICE = device


register_session_item(
    "default_device",
    SessionItem(
        setter=_set_default_device, default=__DEFAULT_DEVICE, persistent=False,
    ),
)
