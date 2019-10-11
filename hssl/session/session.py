from functools import partial

from inspect import getmembers

from . import item
from .item.session_item import SessionItem
from ..logging import DEBUG, ERROR, LOGGER, WARNING, log


__all__ = ["Session", "Unset", "get_session"]


_SESSION_STACK = []
_SESSION_STORE = {}


class Unset:
    def __str__(self):
        return "UNSET"


class Session:
    def __init__(self, **kwargs):
        for name in _SESSION_STORE.keys():
            try:
                value = kwargs.pop(name)
            except KeyError:
                value = Unset()
            setattr(self, name, value)
        if len(kwargs) != 0:
            raise ValueError(
                f'invalid session items: {",".join(kwargs.keys())}'
            )
        self._level = -1

    def __enter__(self):
        _SESSION_STACK.append(self)
        for session in _SESSION_STACK:
            session._level += 1
        apply_session(get_session())

    def __exit__(self, err_type, err, tb):
        if err_type is not None:
            if self._level == 0:
                while tb.tb_next is not None:
                    tb = tb.tb_next
                frame = tb.tb_frame
                LOGGER.findCaller = lambda self, stack_info=None, f=frame: (
                    f.f_code.co_filename,
                    f.f_lineno,
                    f.f_code.co_name,
                    None,
                )
                log(ERROR, "session panic! %s", str(err))
                _get("panic")(get_session(), err_type, err, tb)
        else:
            for session in _SESSION_STACK:
                session._level -= 1
            assert self._level == -1
        assert self == _SESSION_STACK.pop()
        apply_session(get_session())

    def __str__(self):
        return (
            "Session {"
            + "; ".join(
                f"{x}={getattr(self, str(x))}" for x in _SESSION_STORE.keys()
            )
            + "}"
        )

    def __iter__(self):
        for key in _SESSION_STORE.keys():
            yield key, getattr(self, key)


def apply_session(session: Session):
    for name, (setter, default) in _SESSION_STORE.items():
        setter(getattr(session, name, default))


def _get(name: str):
    if name not in _SESSION_STORE.keys():
        raise ValueError(f"{name} is not a session item")

    for obj in reversed(_SESSION_STACK):
        try:
            val = getattr(obj, name)
            if not isinstance(val, Unset):
                return val
        except AttributeError:
            log(WARNING, 'session object lacks attribute "%s"', name)

    _, default = _SESSION_STORE[name]
    return default


def get_session():
    return Session(**{name: _get(name) for name in _SESSION_STORE.keys()})


def _register_session_item(name: str, x: item.SessionItem):
    log(DEBUG, 'registering session item "%s"', name)
    _SESSION_STORE[name] = x
    method_name = f"get_{name}"
    globals()[method_name] = partial(_get, name)
    __all__.append(method_name)


_register_session_item(
    "panic", item.SessionItem(lambda _: None, lambda *_: None)
)

for name, x in getmembers(item, lambda x: isinstance(x, SessionItem)):
    _register_session_item(name, x)
