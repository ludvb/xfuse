from .. import SessionItem, register_session_item

register_session_item(
    "model", SessionItem(setter=lambda _: None, default=None, persistent=True)
)
