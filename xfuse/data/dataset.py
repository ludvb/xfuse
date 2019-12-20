from functools import reduce
from typing import Dict, List, NamedTuple, Optional
from warnings import warn

import numpy as np
import pandas as pd
import torch

from .slide import Slide

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

    def __init__(
        self,
        data: Data,
        genes: Optional[List[str]] = None,
        unify_genes: bool = False,
    ):
        self._data = data
        if genes is None:
            genes = list(
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
        elif not unify_genes:
            warn(
                UserWarning(
                    " ".join(
                        [
                            "Passing a list of `genes` to `Dataset` implies"
                            " `unify_genes=True`.",
                            "Set `unify_genes=True` explicitly to disable this"
                            " warning.",
                        ]
                    )
                )
            )
            unify_genes = True
        self.__genes = genes
        for slide in self.data.slides.values():
            if unify_genes:
                slide.data.genes = self.__genes
            elif slide.data.genes != self.__genes:
                raise ValueError(
                    " ".join(
                        [
                            "Slide uses incongruent set of genes.",
                            "Unify genes by passing `unify_genes=True`.",
                        ]
                    )
                )
        self._data_iterators = {
            name: slide.iterator(slide.data)
            for name, slide in self.data.slides.items()
        }
        self.observations = pd.DataFrame(
            dict(
                type=np.repeat(
                    [x.data.type for x in self._data.slides.values()],
                    [len(x) for x in self._data_iterators.values()],
                ),
                sample=np.repeat(
                    list(self._data_iterators.keys()),
                    [len(x) for x in self._data_iterators.values()],
                ),
                idx=np.concatenate(
                    [range(len(x)) for x in self._data_iterators.values()]
                ),
            )
        )
        self.size = dict(
            zip(*np.unique(self.observations["type"], return_counts=True))
        )

    @property
    def genes(self):
        r"""The genes present in the dataset"""
        return self.__genes

    @property
    def data(self):
        r"""The underlying :class:`Data`"""
        return self._data

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        slide = self.observations["sample"].iloc[idx]
        return dict(
            type=self._data.slides[slide].data.type,
            slide_name=slide,
            **self._data_iterators[slide].__getitem__(
                self.observations["idx"].iloc[idx]
            ),
            effects=torch.as_tensor(self._data.design[slide].values),
        )

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
