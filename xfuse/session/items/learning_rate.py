from .. import SessionItem, register_session_item
from ...logging import DEBUG, log
from ...utility.state.state import get_state_dict, load_state_dict


def _set_learning_rate(learning_rate):
    log(DEBUG, "Setting learning rate to %f", learning_rate)
    load_state_dict(get_state_dict())


register_session_item(
    "learning_rate",
    SessionItem(setter=_set_learning_rate, default=1e-3, persistent=False),
)
