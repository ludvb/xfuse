import torch as t

from .session_item import SessionItem


default_device = SessionItem(
    setter=lambda _: None,
    default=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
)
