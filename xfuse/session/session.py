import warnings
from traceback import format_exc
from typing import Any, Dict, List

from ..logging import DEBUG, ERROR, log
from .session_item import SessionItem

__all__ = [
    "Session",
    "Unset",
    "get_session",
    "get",
    "register_session_item",
    "require",
]


class Unset:
    r"""Marker for unset :class:`Session` items"""

    def __str__(self):
        return "UNSET"


class Session:
    r"""Session context manager"""

    def __init__(self, **kwargs):
        for name in _SESSION_STORE:
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
        _apply_session(get_session())

    def __exit__(self, err_type, err, tb):
        if err_type is not None:
            if self._level == 0:
                log(
                    ERROR,
                    "%s: %s\n%s",
                    err_type.__name__,
                    str(err),
                    format_exc(),
                )
                panic_handler = get("panic")
                if not isinstance(panic_handler, Unset):
                    panic_handler(get_session(), err_type, err, tb)
        else:
            for session in _SESSION_STACK:
                session._level -= 1
            assert self._level == -1
        assert self == _SESSION_STACK.pop()
        _apply_session(get_session())

    def __str__(self):
        return (
            "Session {"
            + "; ".join(f"{x}={getattr(self, str(x))}" for x in _SESSION_STORE)
            + "}"
        )

    def __iter__(self):
        for key in _SESSION_STORE:
            yield key, getattr(self, key)


_SESSION_STACK: List[Session] = []
_SESSION_STORE: Dict[str, SessionItem] = {}


def _apply_session(session: Session):
    for name, (setter, default, _persistent) in _SESSION_STORE.items():
        setter(getattr(session, name, default))


def get(name: str) -> Any:
    r"""
    Gets session item from the current context. Returns its default value if
    unset.
    """
    try:
        return require(name)
    except RuntimeError:
        return _SESSION_STORE[name].default


def require(name: str) -> Any:
    r"""
    Gets session item from the current context. Raises `RuntimeError` if unset.
    """
    if name not in _SESSION_STORE:
        raise ValueError(f"{name} is not a session item")

    for obj in reversed(_SESSION_STACK):
        try:
            val = getattr(obj, name)
            if not isinstance(val, Unset):
                return val
        except AttributeError:
            warnings.warn(f'Session object lacks attribute "{name}"')

    raise RuntimeError(f"Session item {name} has not been set!")


def get_session():
    r"""
    Constructs a new :class:`Sessions` based on the current session context
    """
    return Session(**{name: get(name) for name in _SESSION_STORE})


def register_session_item(name: str, x: SessionItem) -> None:
    r"""Registers new :class:`SessionItem`"""
    log(DEBUG, 'Registering session item "%s"', name)
    _SESSION_STORE[name] = x


register_session_item(
    "panic", SessionItem(lambda _: None, lambda *_: None, persistent=False)
)
