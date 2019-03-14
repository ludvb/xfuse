from functools import WRAPPER_ASSIGNMENTS


class WithInitArgs:
    def __init__(self, *args, **kwargs):
        self._init_args = dict(args=args, kwargs=kwargs)
        super().__init__(*args, **kwargs)

    @property
    def init_args(self):
        return self._init_args


def store_init_args(cls):
    class Wrapped(WithInitArgs, cls):
        pass

    for x in WRAPPER_ASSIGNMENTS:
        try:
            setattr(Wrapped, x, getattr(cls, x))
        except AttributeError:
            pass

    return Wrapped
