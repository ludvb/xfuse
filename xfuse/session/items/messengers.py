from typing import List

import pyro.poutine.runtime as pyro_runtime
from pyro.poutine.messenger import Messenger

from .. import SessionItem, register_session_item


__all__: List[str] = []

__INSTALLED_MESSENGERS: List[Messenger] = []


def _messengers_setter(messengers: List[Messenger]) -> None:
    # pylint: disable=protected-access
    # ^ HACK: This setter installs and uninstall Messengers manually by
    #         modifying Pyro's runtime stack, effectively replacing their
    #         __enter__ and __exit__ methods.
    #         Messengers are always added to the bottom of the stack to avoid
    #         interfering with other Messengers.
    pyro_runtime._PYRO_STACK[:] = [
        messenger
        for messenger in pyro_runtime._PYRO_STACK
        if messenger not in __INSTALLED_MESSENGERS
    ]
    pyro_runtime._PYRO_STACK[:] = [*messengers, *pyro_runtime._PYRO_STACK]
    __INSTALLED_MESSENGERS[:] = messengers


register_session_item(
    "messengers",
    SessionItem(setter=_messengers_setter, default=[], persistent=False),
)
