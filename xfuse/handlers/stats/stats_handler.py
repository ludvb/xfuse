import re
from abc import ABCMeta, abstractmethod
from functools import wraps

from typing import Callable, Optional

from pyro.poutine.messenger import Messenger

from ...logging import DEBUG, log
from ...session import get
from .writer import StatsWriter


class StatsHandler(Messenger, metaclass=ABCMeta):
    r"""Abstract class for stats trackers"""

    def __init__(
        self, predicate: Optional[Callable[..., bool]] = None,
    ):
        super().__init__()

        if predicate is None:
            predicate = lambda **_: True
        self.predicate = predicate

        def _add_writer_method(name, method):
            @wraps(method, assigned=("__doc__", "__annotations__"), updated=())
            def _wrapped(*args, **kwargs):
                stats_writers = get("stats_writers")
                for stats_writer in stats_writers:
                    getattr(stats_writer, name)(*args, **kwargs)

            setattr(self, name, _wrapped)

        for name, method in (
            (name, attr)
            for name in dir(StatsWriter)
            if re.match(r"^add_.*$", name)
            for attr in [getattr(StatsWriter, name)]
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
