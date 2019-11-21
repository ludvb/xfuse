from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix


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

    def __init__(self, data: h5py.File):
        self._data = data
        self._counts = csr_matrix(
            (
                self._data["counts"]["data"],
                self._data["counts"]["indices"],
                self._data["counts"]["indptr"],
            )
        )
        self.H, self.W, _ = self._data["image"].shape

    @property
    def type(self) -> str:
        return "ST"

    @property
    def counts(self):
        r"""Getter for the count data"""
        return self._counts

    @property
    def image(self):
        r"""Getter for the slide image"""
        return self._data["image"]

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
        return self._data["label"]

    def prepare_data(self, image, label):
        r"""Prepare data from image and label patches"""

        labels = np.sort(np.unique(label[label != 0]))
        data = self._counts[(labels - 1).tolist()]
        label = np.searchsorted([0, *labels], label)

        return dict(
            image=(
                torch.as_tensor(image / 255 * 2 - 1).permute(2, 0, 1).float()
            ),
            label=torch.as_tensor(label).long(),
            data=torch.as_tensor(data.todense()).float(),
        )


class Slide(NamedTuple):
    r"""Data structure for tissue slide"""
    data: SlideData
    iterator: Callable[[SlideData], SlideIterator]
