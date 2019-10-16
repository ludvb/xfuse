from .. import SessionItem, register_session_item


class GlobalStep:
    def __init__(self, value=0):
        self.value = 0

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
