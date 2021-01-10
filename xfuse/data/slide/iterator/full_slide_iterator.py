from ..data import AnnotatedImage, SlideData, STSlide
from . import SlideIterator


class FullSlideIterator(SlideIterator):
    r"""A :class:`SlideIterator` that yields the full (uncropped) sample"""

    def __init__(self, slide: SlideData):
        self._slide = slide

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if isinstance(self._slide, STSlide):
            image = self._slide.image[()].transpose(2, 0, 1)
            label = self._slide.label[()]
            return self._slide.prepare_data(image, label)
        if isinstance(self._slide, AnnotatedImage):
            return {
                "image": self._slide.image.permute(2, 0, 1),
                "label": self._slide.label,
                "name": self._slide.name,
                "label_names": self._slide.label_names,
            }
        raise NotImplementedError()
