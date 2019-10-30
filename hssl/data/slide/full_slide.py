from .slide import SlideIterator, STSlide

__all__ = ["FullSlide"]


class FullSlide(SlideIterator):
    r"""A :class:`SlideIterator` that yields the full (uncropped) sample"""

    def __init__(self, slide: STSlide):
        self._slide = slide

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = self._slide.image.to_array()
        label = self._slide.label.to_array().squeeze()
        return self._slide.prepare_data(image, label)
