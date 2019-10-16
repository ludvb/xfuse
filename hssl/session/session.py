from typing import Any, Dict, List

from ..logging import DEBUG, ERROR, LOGGER, WARNING, log
from .session_item import SessionItem

__all__ = ["Session", "Unset", "get_session", "get", "register_session_item"]


class Unset:
    def __str__(self):
        return "UNSET"


class Session:
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
                get("panic")(get_session(), err_type, err, tb)
        else:
            for session in _SESSION_STACK:
                session._level -= 1
            assert self._level == -1
        assert self == _SESSION_STACK.pop()
        apply_session(get_session())

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


def apply_session(session: Session):
    for name, (setter, default) in _SESSION_STORE.items():
        setter(getattr(session, name, default))


def get(name: str) -> Any:
    if name not in _SESSION_STORE:
        raise ValueError(f"{name} is not a session item")

    for obj in reversed(_SESSION_STACK):
        try:
            val = getattr(obj, name)
            if not isinstance(val, Unset):
                return val
        except AttributeError:
            log(WARNING, 'session object lacks attribute "%s"', name)

    return _SESSION_STORE[name].default


def get_session():
    return Session(**{name: get(name) for name in _SESSION_STORE})


def register_session_item(name: str, x: SessionItem) -> None:
    log(DEBUG, 'registering session item "%s"', name)
    _SESSION_STORE[name] = x


register_session_item("panic", SessionItem(lambda _: None, lambda *_: None))
