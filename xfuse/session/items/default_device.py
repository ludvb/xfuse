import torch as t

from .. import SessionItem, register_session_item

register_session_item(
    "default_device",
    SessionItem(
        setter=lambda _: None,
        default=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
    ),
)
