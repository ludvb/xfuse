from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pyvips
from imageio import imread

__all__ = ["Image", "LazyImage", "PreloadedImage"]


class Image(ABC):
    r"""Abstract class for image data"""

    @property
    @abstractmethod
    def height(self) -> int:
        r"""Image height"""

    @property
    @abstractmethod
    def width(self) -> int:
        r"""Image width"""

    @property
    @abstractmethod
    def channels(self) -> int:
        r"""Image channels"""

    @abstractmethod
    def extract(self, x: int, y: int, width: int, height: int) -> Image:
        r"""Extract image region"""

    @abstractmethod
    def to_array(self) -> np.ndarray:
        r"""Convert the image to a :class:`numpy.ndarray`"""


class LazyImage(Image):
    r"""Image data that is read from disk when needed"""

    def __init__(self, image: pyvips.Image):
        self._image = image

    @staticmethod
    def from_file(image_file: str) -> LazyImage:
        r"""Creates a :class:`LazyImage` from a file"""
        return LazyImage(pyvips.Image.new_from_file(image_file))

    @property
    def height(self):
        return self._image.height

    @property
    def width(self):
        return self._image.width

    @property
    def channels(self):
        return self._image.bands

    def extract(self, x: int, y: int, width: int, height: int) -> LazyImage:
        return LazyImage(self._image.extract_area(x, y, width, height))

    def to_array(self) -> np.ndarray:
        return np.ndarray(
            buffer=self._image.write_to_memory(),
            shape=(self.height, self.width, self.channels),
            dtype=(
                {
                    "uchar": np.uint8,
                    "char": np.int8,
                    "ushort": np.uint16,
                    "short": np.int16,
                    "uint": np.uint32,
                    "int": np.int32,
                    "float": np.float32,
                    "double": np.float64,
                    "complex": np.complex64,
                    "dpcomplex": np.complex128,
                }[self._image.format]
            ),
        )


class PreloadedImage(Image):
    r"""Image data that is stored in RAM"""

    def __init__(self, image: np.ndarray):
        if image.ndim not in [2, 3]:
            raise ValueError("`image` must be of shape HW or HWC")
        if image.ndim == 2:
            image = image[..., None]
        self._image = image

    @staticmethod
    def from_file(image_file: str) -> PreloadedImage:
        r"""Creates a :class:`PreloadedImage` from a file"""
        return PreloadedImage(np.array(imread(image_file)))

    @property
    def height(self):
        # pylint: disable=unsubscriptable-object
        return self._image.shape[0]

    @property
    def width(self):
        # pylint: disable=unsubscriptable-object
        return self._image.shape[1]

    @property
    def channels(self):
        # pylint: disable=unsubscriptable-object
        return self._image.shape[2]

    def extract(
        self, x: int, y: int, width: int, height: int
    ) -> PreloadedImage:
        return PreloadedImage(self._image[y : y + height, x : x + width])

    def to_array(self):
        return self._image
