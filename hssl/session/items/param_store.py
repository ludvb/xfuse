# pylint: disable=protected-access

import pyro

from .. import SessionItem, register_session_item

_DEFAULT_STORE = pyro.poutine.runtime._PYRO_PARAM_STORE


def _setter(store):
    pyro.poutine.runtime._PYRO_PARAM_STORE = store
    pyro.primitives._PYRO_PARAM_STORE = store
    pyro.primitives._param = pyro.poutine.runtime.effectful(
        store.get_param, type="param"
    )


register_session_item(
    "param_store", SessionItem(setter=_setter, default=_DEFAULT_STORE)
)
