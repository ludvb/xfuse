from abc import abstractmethod

from typing import Callable, Optional

from pyro.poutine.messenger import Messenger

from torch.utils.tensorboard.writer import SummaryWriter

from ...session import get_global_step


class StatsHandler(Messenger):
    def __init__(
            self,
            writer: SummaryWriter,
            predicate: Optional[Callable[..., bool]] = None,
    ):
        from functools import wraps
        import inspect

        if predicate is None:
            predicate = lambda **_: True
        self.predicate = predicate

        self.__writer = writer

        def _add_writer_method(name, method):
            setattr(
                self,
                name,
                wraps(method)(
                    lambda *args, **kwargs: method(
                        *args, global_step=int(get_global_step()), **kwargs)
                    if "global_step" in inspect.signature(method).parameters
                    else
                    lambda *args, **kwargs: method(*args, **kwargs)
                ),
            )

        for name, method in (
                (name, attr)
                for name, attr in (
                        (name, getattr(self.__writer, name))
                        for name in dir(self.__writer)
                )
                if name[0] != '_' if callable(attr)
        ):
            _add_writer_method(name, method)

    @abstractmethod
    def _handle(self, **msg) -> None:
        pass

    @abstractmethod
    def _select_msg(self, **msg) -> bool:
        pass

    def _pyro_post_sample(self, msg):
        if self._select_msg(**msg) and self.predicate(**msg):
            self._handle(**msg)
