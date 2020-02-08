from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Callable, List, NamedTuple

import h5py
import numpy as np
import scipy.sparse
import torch

from ...logging import DEBUG, log


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

    @abstractproperty
    def type(self) -> str:
        r"""The type tag of this slide"""

    @abstractproperty
    def genes(self) -> List[str]:
        r"""Genes returned from this dataset"""

    @genes.setter
    def genes(self, genes: List[str]) -> SlideData:
        r"""Setter for which genes to return from this dataset"""

    @abstractmethod
    def annotation(self, name) -> torch.Tensor:
        r"""Get annotation layer by name"""


class STSlide(SlideData):
    r""":class:`SlideData` for Spatial Transcriptomics slides"""

    def __init__(
        self,
        data: h5py.File,
        cache_data: bool = True,
        min_counts: float = 0,
        always_filter: List[int] = None,
        always_keep: List[int] = None,
    ):
        self._data = data
        if cache_data:
            self._counts = self.__construct_count_matrix()
            self._label = self._data["label"][()]
            self._image = self._data["image"][()]
        else:
            self._counts = None
            self._label = None
            self._image = None
        self.genes = list(self._data["counts"]["columns"][()])
        self.__always_filter = always_filter or []
        self.__always_keep = always_keep or []
        self.min_counts = min_counts
        self.H, self.W, _ = self._data["image"].shape

    @property
    def type(self) -> str:
        return "ST"

    @property
    def min_counts(self) -> float:
        r"""
        The minimum number of reads for an ST spot to be included in this
        dataset. This attribute can be used to filter out low quality spots.
        """
        return self.__min_counts

    @min_counts.setter
    def min_counts(self, n: float):
        self.__min_counts = n
        self.__label_mask = np.unique(
            self.__always_filter
            + [
                x
                for x in (
                    (np.array(self.counts.sum(1)).flatten() < n).nonzero()[0]
                    + 1
                )
                if x not in self.__always_keep
            ]
        )
        if self.__label_mask.shape[0] > 0:
            log(
                DEBUG,
                "The following labels will be masked out in %s: %s",
                self._data.filename,
                ", ".join(map(str, self.__label_mask)),
            )

    @property
    def genes(self):
        return list(self.__gene_list.copy())

    @genes.setter
    def genes(self, genes: List[str]) -> STSlide:
        self.__gene_list = np.array(genes)
        idxs = {
            gene: i
            for i, gene in enumerate(self._data["counts"]["columns"][()])
        }
        self.__gene_idxs = np.array(
            [idxs[gene] if gene in idxs else -1 for gene in genes]
        )
        return self

    def __construct_count_matrix(self):
        counts = scipy.sparse.csr_matrix(
            (
                self._data["counts"]["data"],
                self._data["counts"]["indices"],
                self._data["counts"]["indptr"],
            ),
            shape=(
                len(self._data["counts"]["index"]),
                len(self._data["counts"]["columns"]),
            ),
        )
        counts = scipy.sparse.hstack(
            [counts, np.zeros((counts.shape[0], 1))], format="csr"
        )
        return counts

    @property
    def counts(self):
        r"""Getter for the count data"""
        if self._counts is not None:
            counts = self._counts
        else:
            counts = self.__construct_count_matrix()
        return counts[:, self.__gene_idxs]

    @property
    def image(self):
        r"""Getter for the slide image"""
        if self._image is not None:
            return self._image
        return self._data["image"]

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
        if self._label is not None:
            return self._label
        return self._data["label"]

    def annotation(self, name):
        if name not in self._data["annotation"]:
            raise RuntimeError(f'annotation layer "{name}" is missing')
        return self._data["annotation"][name][()]

    def prepare_data(self, image, label):
        r"""Prepare data from image and label patches"""

        label[np.isin(label, self.__label_mask)] = 0
        labels = np.sort(np.unique(label[label != 0]))
        data = self.counts[(labels - 1).tolist()]
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
