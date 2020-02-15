from .. import SessionItem, register_session_item

register_session_item(
    "optimizer",
    SessionItem(setter=lambda _: None, default=None, persistent=False),
)
