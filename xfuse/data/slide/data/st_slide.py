from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse
import torch

import h5py

from ....logging import DEBUG, log
from .slide_data import SlideData


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
            self.genes = list(data["counts"]["columns"][()].astype(str))
        self.__always_filter = always_filter or []
        self.__always_keep = always_keep or []
        self.cache_data = cache_data
        self._counts = None
        self._label = None
        self._image = None
        self.min_counts = min_counts

    @property
    def data_type(self) -> str:
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
                gene: i
                for i, gene in enumerate(
                    data["counts"]["columns"][()].astype(str)
                )
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

    def annotation(self, name) -> Tuple[np.ndarray, Dict[int, str]]:
        r"""Getter for annotation layers"""
        with h5py.File(self._datafile, "r") as data:
            if name not in data["annotation"]:
                raise RuntimeError(f'Annotation layer "{name}" is missing')
            return (
                data["annotation"][name]["label"][()],
                dict(
                    zip(
                        data["annotation"][name]["names"]["keys"][()],
                        data["annotation"][name]["names"]["values"][()].astype(
                            str
                        ),
                    )
                ),
            )

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
