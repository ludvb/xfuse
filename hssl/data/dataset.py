from functools import reduce
from typing import Dict, List, NamedTuple, Optional

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

    def __init__(self, data: Data, genes: Optional[List[str]] = None):
        self._data = data
        if genes is None:
            genes = list(
                reduce(
                    set.union,  # type: ignore
                    (
                        set(slide.data.genes)
                        for slide in self.data.slides.values()
                    ),
                )
            )
        for slide in self.data.slides.values():
            slide.data.genes = genes
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
    def data(self):
        r"""The underlying :class:`Data`"""
        return self._data

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        slide = self.observations["sample"].iloc[idx]
        return dict(
            type=self._data.slides[slide].data.type,
            **self._data_iterators[slide].__getitem__(
                self.observations["idx"].iloc[idx]
            ),
            effects=torch.as_tensor(self._data.design[slide].values),
        )
