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

    @property
    @abstractmethod
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
        min_counts: float = 0,
        always_filter: List[int] = None,
        always_keep: List[int] = None,
    ):
        self._data = data
        self._counts = scipy.sparse.csr_matrix(
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
        self._counts = scipy.sparse.hstack(
            [self._counts, np.zeros((self._counts.shape[0], 1))], format="csr"
        )
        self.genes = list(self._data["counts"]["columns"][()])
        self.__always_filter = always_filter or []
        self.__always_keep = always_keep or []
        self.min_counts = min_counts
        self.H, self.W, _ = self._data["image"].shape

    @property
    def type(self) -> str:
        return "ST"

    def __recompute_summary_statistics(self):
        try:
            counts = self.counts[
                ~np.isin(np.arange(self.counts.shape[0]), self.__label_mask)
            ]
        except AttributeError:
            counts = self.counts
        counts = counts.todense()
        # pylint: disable=attribute-defined-outside-init
        self.__counts_means = np.mean(np.array(counts), 0)
        self.__counts_stdvs = np.nanstd(np.array(counts), 0)
        return self

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
        self.__recompute_summary_statistics()

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
        self.__recompute_summary_statistics()
        return self

    @property
    def counts(self):
        r"""Getter for the count data"""
        return self._counts[:, self.__gene_idxs]

    @property
    def means(self):
        r"""Count mean for each gene"""
        return self.__counts_means

    @property
    def stdvs(self):
        r"""Count standard deviation for each gene"""
        return self.__counts_stdvs

    @property
    def image(self):
        r"""Getter for the slide image"""
        return self._data["image"]

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
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
            means=torch.as_tensor(self.means).float(),
            stdvs=torch.as_tensor(self.stdvs).float(),
        )


class Slide(NamedTuple):
    r"""Data structure for tissue slide"""
    data: SlideData
    iterator: Callable[[SlideData], SlideIterator]
