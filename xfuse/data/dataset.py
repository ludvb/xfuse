from functools import reduce
from typing import Dict, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch

from .slide import Slide
from ..session import get

__all__ = ["Data", "Dataset"]


class Data(NamedTuple):
    r"""Dataset consisting of multiple instances of :class:`Slide`"""

    slides: Dict[str, Slide]
    design: pd.DataFrame


class Dataset(torch.utils.data.Dataset):
    r"""
    :class:`~torch.utils.data.Dataset` yielding data from a :class:`Data`
    instance
    """

    def __init__(self, data: Data):
        self._data = data

        if get("genes"):
            self.genes = get("genes")
        else:
            self.genes = list(
                sorted(
                    reduce(
                        set.union,  # type: ignore
                        (
                            set(slide.data.genes)
                            for slide in self.data.slides.values()
                        ),
                    )
                )
            )

        self._data_iterators = {
            name: slide.iterator(slide.data)
            for name, slide in self.data.slides.items()
        }
        self.observations = pd.DataFrame(
            dict(
                data_type=np.repeat(
                    [x.data.data_type for x in self._data.slides.values()],
                    [len(x) for x in self._data_iterators.values()],
                ),
                slide=np.repeat(
                    list(self._data_iterators.keys()),
                    [len(x) for x in self._data_iterators.values()],
                ),
                idx=np.concatenate(
                    [range(len(x)) for x in self._data_iterators.values()]
                ),
            )
        )

    def size(
        self,
        data_type: Optional[str] = None,
        slide: Optional[str] = None,
        covariate: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> int:
        """Returns the size of the dataset for a given `data_type`"""
        observations = self.observations
        if data_type is not None:
            observations = observations[observations.data_type == data_type]
        if slide is not None:
            observations = observations[observations.slide == slide]
        if covariate is not None:
            observations = observations.merge(
                self.data.design[covariate].rename("condition"),
                left_on="slide",
                right_index=True,
            )
            if condition is not None:
                observations = observations[
                    observations["condition"] == condition
                ]
        return len(observations)

    @property
    def genes(self):
        r"""The genes present in the dataset"""
        return self.__genes

    @genes.setter
    def genes(self, genes):
        self.__genes = genes
        for slide in self.data.slides.values():
            slide.data.genes = genes

    @property
    def data(self):
        r"""The underlying :class:`Data`"""
        return self._data

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        slide = self.observations["slide"].iloc[idx]
        return dict(
            data_type=self._data.slides[slide].data.data_type,
            slide=slide,
            covariates=dict(self.data.design.loc[slide].iteritems()),
            **self._data_iterators[slide].__getitem__(
                self.observations["idx"].iloc[idx]
            ),
        )

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
