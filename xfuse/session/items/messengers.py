from contextlib import ExitStack
from typing import List

from pyro.poutine.messenger import Messenger

from .. import SessionItem, register_session_item


__all__: List[str] = []


__STACK = ExitStack()


def _messengers_setter(messengers: List[Messenger]) -> None:
    __STACK.close()
    for messenger in messengers:
        __STACK.enter_context(messenger)


register_session_item(
    "messengers",
    SessionItem(setter=_messengers_setter, default=[], persistent=False),
)
