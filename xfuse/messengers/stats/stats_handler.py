from abc import ABCMeta, abstractmethod
from io import BytesIO
from typing import Callable, List, Optional

import matplotlib
import torch
from imageio import imread
from pyro.poutine.messenger import Messenger

from ...logging import DEBUG, log
from ...session import Session, get
from ...utility.file import chdir
from .writer import StatsWriter


__all__ = [
    "StatsHandler",
    "log_figure",
    "log_histogram",
    "log_image",
    "log_images",
    "log_scalar",
]


class StatsHandler(Messenger, metaclass=ABCMeta):
    r"""Abstract class for stats trackers"""

    def __init__(
        self, predicate: Optional[Callable[..., bool]] = None,
    ):
        super().__init__()

        if predicate is None:
            predicate = lambda **_: not get("eval")
        self.predicate = predicate

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


def log_figure(tag: str, figure: matplotlib.figure.Figure, **kwargs,) -> None:
    r"""
    Converts :class:`~matplotlib.figure.Figure`` to image data and logs it
    using :func:`log_image`
    """
    if "format" not in kwargs:
        kwargs["format"] = "tiff"
    bio = BytesIO()
    with Session(mpl_backend="Agg"):
        figure.savefig(bio, **kwargs)
    bio.seek(0)
    fig_image = torch.as_tensor(imread(bio))
    log_image(tag, img_tensor=fig_image)


def log_histogram(*args, **kwargs) -> None:
    r"""Pushes histogram data to the session `stats_writers`"""
    stats_writers: List[StatsWriter] = get("stats_writers")
    with chdir("/stats"), torch.no_grad():
        for stats_writer in stats_writers:
            stats_writer.write_histogram(*args, **kwargs)


def log_image(*args, **kwargs) -> None:
    r"""Pushes image data to the session `stats_writers`"""
    stats_writers: List[StatsWriter] = get("stats_writers")
    with chdir("/stats"), torch.no_grad():
        for stats_writer in stats_writers:
            stats_writer.write_image(*args, **kwargs)


def log_images(*args, **kwargs) -> None:
    r"""Pushes image grid to the session `stats_writers`"""
    stats_writers: List[StatsWriter] = get("stats_writers")
    with chdir("/stats"), torch.no_grad():
        for stats_writer in stats_writers:
            stats_writer.write_images(*args, **kwargs)


def log_scalar(*args, **kwargs) -> None:
    r"""Pushes scalar to the session `stats_writers`"""
    stats_writers: List[StatsWriter] = get("stats_writers")
    with chdir("/stats"), torch.no_grad():
        for stats_writer in stats_writers:
            stats_writer.write_scalar(*args, **kwargs)


def log_scalars(*args, **kwargs) -> None:
    r"""Pushes scalars to the session `stats_writers`"""
    stats_writers: List[StatsWriter] = get("stats_writers")
    with chdir("/stats"), torch.no_grad():
        for stats_writer in stats_writers:
            stats_writer.write_scalars(*args, **kwargs)
