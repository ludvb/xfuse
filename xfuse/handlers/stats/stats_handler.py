from abc import ABC, abstractmethod

from functools import wraps
import inspect

from typing import Callable, Optional

from pyro.poutine.messenger import Messenger

from torch.utils.tensorboard.writer import SummaryWriter

from ...logging import DEBUG, log
from ...session import get


class StatsHandler(ABC, Messenger):
    r"""Abstract class for stats trackers"""

    def __init__(
        self,
        writer: SummaryWriter,
        predicate: Optional[Callable[..., bool]] = None,
    ):
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
                        *args, global_step=get("training_data").step, **kwargs
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
        log(DEBUG, "Activating stats tracker: %s", type(self).__name__)
        super().__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # pylint: disable=arguments-differ
        log(DEBUG, "Deactivating stats tracker: %s", type(self).__name__)
        super().__exit__(*args, **kwargs)

    @abstractmethod
    def _handle(self, **msg) -> None:
        pass

    @abstractmethod
    def _select_msg(self, **msg) -> bool:
        pass

    def _postprocess_message(self, msg):
        if self._select_msg(**msg) and self.predicate(**msg):
            self._handle(**msg)
