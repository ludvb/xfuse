from .session_item import SessionItem


class GlobalStep:
    def __init__(self, value=0):
        self.value = 0

    def __iadd__(self, n):
        self.value += n
        return self

    def __str__(self):
        return f'GlobalStep({str(self.value)})'

    def __int__(self):
        return self.value


global_step = SessionItem(setter=lambda _: None, default=GlobalStep())
