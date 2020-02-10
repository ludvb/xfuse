from .. import SessionItem, register_session_item

register_session_item(
    "dataloader",
    SessionItem(setter=lambda _: None, default=None, persistent=False),
)
