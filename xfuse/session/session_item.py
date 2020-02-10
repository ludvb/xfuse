from typing import Any, Callable, NamedTuple


class SessionItem(NamedTuple):
    r"""Data structure for session items"""

    setter: Callable[[Any], None]
    default: Any
    persistent: bool = True
