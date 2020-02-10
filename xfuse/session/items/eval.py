from .. import SessionItem, register_session_item


register_session_item(
    "eval", SessionItem(setter=lambda _: None, default=False, persistent=False)
)
