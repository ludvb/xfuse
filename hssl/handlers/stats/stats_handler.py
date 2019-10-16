from abc import ABC, abstractmethod

from typing import Callable, Optional

from pyro.poutine.messenger import Messenger

from torch.utils.tensorboard.writer import SummaryWriter

from ...logging import DEBUG, log
from ...session import get


class StatsHandler(ABC, Messenger):
    def __init__(
        self,
        writer: SummaryWriter,
        predicate: Optional[Callable[..., bool]] = None,
    ):
        from functools import wraps
        import inspect

        super().__init__()

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
                        *args, global_step=int(get("global_step")), **kwargs
                    )
                    if "global_step" in inspect.signature(method).parameters
                    else method
                ),
            )

        for name, method in (
            (name, attr)
            for name, attr in (
                (name, getattr(self.__writer, name))
                for name in dir(self.__writer)
            )
            if name[0] != "_"
            if callable(attr)
        ):
            _add_writer_method(name, method)

    def __enter__(self, *args, **kwargs):
        # pylint: disable=arguments-differ
        log(DEBUG, "activating stats tracker: %s", type(self).__name__)
        super().__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # pylint: disable=arguments-differ
        log(DEBUG, "deactivating stats tracker: %s", type(self).__name__)
        super().__exit__(*args, **kwargs)

    @abstractmethod
    def _handle(self, **msg) -> None:
        pass

    @abstractmethod
    def _select_msg(self, **msg) -> bool:
        pass

    def _pyro_post_sample(self, msg):
        if self._select_msg(**msg) and self.predicate(**msg):
            self._handle(**msg)
