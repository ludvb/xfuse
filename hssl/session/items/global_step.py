from .. import SessionItem, register_session_item


class GlobalStep:
    r"""Holds the value for the global step"""

    def __init__(self, value=0):
        self.value = value

    def __iadd__(self, n):
        self.value += n
        return self

    def __str__(self):
        return f"GlobalStep({str(self.value)})"

    def __int__(self):
        return self.value


register_session_item(
    "global_step", SessionItem(setter=lambda _: None, default=GlobalStep())
)
