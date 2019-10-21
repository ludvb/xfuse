# pylint: disable=protected-access

import pyro.poutine as p

from .. import SessionItem, register_session_item

_DEFAULT_PYRO_STACK = p.runtime._PYRO_STACK


def _setter(pyro_stack):
    p.runtime._PYRO_STACK = pyro_stack
    p.messenger._PYRO_STACK = pyro_stack


register_session_item(
    "pyro_stack", SessionItem(setter=_setter, default=_DEFAULT_PYRO_STACK)
)
