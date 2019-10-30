from abc import ABCMeta, abstractmethod
from typing import NamedTuple

import numpy as np
import torch

from ..image import Image


class SlideIterator(metaclass=ABCMeta):
    r"""Slide iterator"""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)


class SlideData(metaclass=ABCMeta):
    r"""Abstract class for different kinds of slide data"""

    @property
    @abstractmethod
    def type(self) -> str:
        r"""The type tag of this slide"""


class STSlide(SlideData):
    r""":class:`SlideData` for Spatial Transcriptomics slides"""

    def __init__(self, data: torch.Tensor, image: Image, label: Image):
        if data.is_sparse:  # type: ignore
            data = data.to_dense()

        self._image = image
        self._label = label
        self._data = data

        self.H, self.W = self._image.height, self._image.width
        assert self.H == self._label.height and self.W == self._label.width

    @property
    def type(self) -> str:
        return "ST"

    @property
    def data(self):
        r"""Getter for the count data"""
        return self._data

    @property
    def image(self):
        r"""Getter for the slide image"""
        return self._image

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
        return self._label

    def prepare_data(self, image, label):
        r"""Prepare data from image and label patches"""

        labels = np.sort(np.unique(label[label != 0]))
        data = self._data[(labels - 1).tolist()]
        label = np.searchsorted([0, *labels], label)

        return dict(
            image=(
                torch.as_tensor(image / 255 * 2 - 1).permute(2, 0, 1).float()
            ),
            label=torch.as_tensor(label).long(),
            data=data,
        )


class Slide(NamedTuple):
    r"""Data structure for tissue slide"""
    data: SlideData
    iterator: SlideIterator
