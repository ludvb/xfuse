from .. import SessionItem, register_session_item
from ...logging import INFO, set_level


register_session_item(
    "log_level", SessionItem(setter=set_level, default=INFO, persistent=False)
)
