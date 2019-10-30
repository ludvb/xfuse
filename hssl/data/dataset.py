from typing import List, NamedTuple

import numpy as np
import pandas as pd
import torch

from .slide import Slide

__all__ = ["Data", "Dataset"]


class Data(NamedTuple):
    r"""Dataset consisting of multiple instances of :class:`Slide`"""

    slides: List[Slide]
    design: pd.DataFrame


class Dataset(torch.utils.data.Dataset):
    r"""
    :class:`~torch.utils.data.Dataset` yielding data from a :class:`Data`
    instance
    """

    def __init__(self, data: Data):
        self._data = data
        self._data_iterators = [
            slide.iterator(slide.data) for slide in self.data.slides
        ]
        self.observations = pd.DataFrame(
            dict(
                type=np.repeat(
                    [x.data.type for x in self._data.slides],
                    [len(x) for x in self._data_iterators],
                ),
                sample=np.repeat(
                    range(len(self._data_iterators)),
                    [len(x) for x in self._data_iterators],
                ),
                idx=np.concatenate(
                    [range(len(x)) for x in self._data_iterators]
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
