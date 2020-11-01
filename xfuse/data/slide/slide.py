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
        datafile: str,
        cache_data: bool = True,
        min_counts: float = 0,
        always_filter: List[int] = None,
        always_keep: List[int] = None,
    ):
        self._datafile = datafile
        with h5py.File(datafile, "r") as data:
            self.H, self.W, _ = data["image"].shape
            self.genes = list(data["counts"]["columns"][()])
        self.__always_filter = always_filter or []
        self.__always_keep = always_keep or []
        self.cache_data = cache_data
        self._counts = None
        self._label = None
        self._image = None
        self.min_counts = min_counts

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
                self._datafile,
                ", ".join(map(str, self.__label_mask)),
            )

    @property
    def genes(self):
        return list(self.__gene_list.copy())

    @genes.setter
    def genes(self, genes: List[str]) -> STSlide:
        self.__gene_list = np.array(genes)
        with h5py.File(self._datafile, "r") as data:
            idxs = {
                gene: i for i, gene in enumerate(data["counts"]["columns"][()])
            }
        self.__gene_idxs = np.array(
            [idxs[gene] if gene in idxs else -1 for gene in genes]
        )
        self._counts = None
        return self

    def __construct_count_matrix(self):
        with h5py.File(self._datafile, "r") as data:
            counts = scipy.sparse.csr_matrix(
                (
                    data["counts"]["data"],
                    data["counts"]["indices"],
                    data["counts"]["indptr"],
                ),
                shape=(
                    len(data["counts"]["index"]),
                    len(data["counts"]["columns"]),
                ),
            )
        counts = scipy.sparse.hstack(
            [counts, np.zeros((counts.shape[0], 1))], format="csr"
        )
        counts = counts[:, self.__gene_idxs]
        return counts

    @property
    def counts(self):
        r"""Getter for the count data"""
        if self._counts is not None:
            return self._counts
        counts = self.__construct_count_matrix()
        if self.cache_data:
            self._counts = counts
        return counts

    @property
    def image(self):
        r"""Getter for the slide image"""
        if self._image is not None:
            return self._image
        data = h5py.File(self._datafile, "r")
        image = data["image"]
        if self.cache_data:
            self._image = image[()]
        return image

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
        if self._label is not None:
            return self._label
        data = h5py.File(self._datafile, "r")
        label = data["label"]
        if self.cache_data:
            self._label = label[()]
        return label

    def annotation(self, name):
        with h5py.File(self._datafile, "r") as data:
            if name not in data["annotation"]:
                raise RuntimeError(f'Annotation layer "{name}" is missing')
            return data["annotation"][name][()]

    def prepare_data(self, image, label):
        r"""Prepare data from image and label patches"""

        label[np.isin(label, self.__label_mask)] = 0
        labels = np.sort(np.unique(label[label != 0]))
        data = self.counts[(labels - 1).tolist()]
        label = np.searchsorted([0, *labels], label)

        return dict(
            image=torch.as_tensor(image).float(),
            label=torch.as_tensor(label).long(),
            data=torch.as_tensor(data.todense()).float(),
        )


class Slide(NamedTuple):
    r"""Data structure for tissue slide"""
    data: SlideData
    iterator: Callable[[SlideData], SlideIterator]
