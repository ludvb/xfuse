from typing import Any, Callable, NamedTuple


class SessionItem(NamedTuple):
    setter: Callable[[Any], None]
    default: Any
