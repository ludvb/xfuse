from .session_item import SessionItem
from ...logging import INFO, set_level


log_level = SessionItem(setter=set_level, default=INFO)
