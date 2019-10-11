from functools import WRAPPER_ASSIGNMENTS

from inspect import signature


def store_init_args(cls):
    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            self._init_args = dict(
                args=args,
                kwargs={
                    **{
                        p.name: p.default
                        for p in list(
                            signature(cls.__init__).parameters.values()
                        )[1:]
                    },
                    **kwargs,
                },
            )
            super().__init__(*args, **kwargs)

        @property
        def init_args(self):
            return self._init_args

    for x in WRAPPER_ASSIGNMENTS:
        try:
            setattr(Wrapped, x, getattr(cls, x))
        except AttributeError:
            pass

    return Wrapped
