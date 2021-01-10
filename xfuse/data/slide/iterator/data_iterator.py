import torch

from . import SlideIterator
from ..data import STSlide


class DataIterator(SlideIterator):
    r"""
    A :class:`SlideIterator` that yields only the count data from the slide
    """

    def __init__(self, slide: STSlide):
        self._slide = slide

    def __len__(self):
        return self._slide.counts.shape[0]

    def __getitem__(self, idx):
        return dict(data=torch.as_tensor(self._slide.counts[idx].todense()))
