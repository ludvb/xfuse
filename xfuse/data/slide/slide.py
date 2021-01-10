from typing import Callable, NamedTuple

from .data import SlideData
from .iterator import SlideIterator


class Slide(NamedTuple):
    r"""Data structure for tissue slide"""
    data: SlideData
    iterator: Callable[[SlideData], SlideIterator]
