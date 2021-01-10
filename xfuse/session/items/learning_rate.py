from .. import SessionItem, register_session_item
from ...logging import DEBUG, log
from ...utility.state.state import get_state_dict, load_state_dict


__DEFAULT_LR = 0.0003
__CURRENT_LR = __DEFAULT_LR


def _set_learning_rate(learning_rate):
    # pylint: disable=global-statement
    global __CURRENT_LR
    if learning_rate != __CURRENT_LR:
        log(DEBUG, "Setting learning rate to %f", learning_rate)
        load_state_dict(get_state_dict())
        __CURRENT_LR = learning_rate


register_session_item(
    "learning_rate",
    SessionItem(setter=_set_learning_rate, default=1e-3, persistent=False),
)
