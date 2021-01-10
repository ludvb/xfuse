from .slide import SlideIterator, SlideData, STSlide, SyntheticSlide
from ...session import get

__all__ = ["FullSlide"]


class FullSlide(SlideIterator):
    r"""A :class:`SlideIterator` that yields the full (uncropped) sample"""

    def __init__(self, slide: SlideData, repeat: int = 1):
        self._slide = slide
        self._size = repeat

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if isinstance(self._slide, STSlide):
            image = self._slide.image[()].transpose(2, 0, 1)
            label = self._slide.label[()]
            return self._slide.prepare_data(image, label)

        if isinstance(self._slide, SyntheticSlide):
            self._slide.reset_data()
            image = self._slide.image.transpose(2, 0, 1)
            label = self._slide.label
            return self._slide.prepare_data(image, label)

        raise NotImplementedError()
