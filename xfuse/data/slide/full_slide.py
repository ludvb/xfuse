from .slide import SlideIterator, SlideData

__all__ = ["FullSlide"]


class FullSlide(SlideIterator):
    r"""A :class:`SlideIterator` that yields the full (uncropped) sample"""

    def __init__(self, slide: SlideData):
        self._slide = slide

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self._slide.type == "ST":
            image = self._slide.image[()].transpose(2, 0, 1)
            label = self._slide.label[()]
            return self._slide.prepare_data(image, label)
        raise NotImplementedError()
