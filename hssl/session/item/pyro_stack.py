import pyro.poutine as p

from .session_item import SessionItem


_DEFAULT_PYRO_STACK = p.runtime._PYRO_STACK


def _setter(pyro_stack):
    p.runtime._PYRO_STACK = pyro_stack
    p.messenger._PYRO_STACK = pyro_stack


pyro_stack = SessionItem(setter=_setter, default=_DEFAULT_PYRO_STACK)
