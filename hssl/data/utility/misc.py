import itertools as it
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate  # type: ignore

from ...session import get
from ..dataset import Dataset

__all__ = ["make_dataloader", "spot_size"]


def spot_size(dataset: Dataset) -> np.float64:
    r"""Computes the median spot size in the :class:`Dataset`"""
    return np.median(
        np.concatenate(
            [
                np.bincount(d["label"].flatten())[1:]
                for d in dataset  # type: ignore
            ]
        )
    )


def make_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader:
    r"""Creates a :class:`~torch.utils.data.DataLoader` for `dataset`"""

    def _collate(xs):
        def _remove_key(v):
            v.pop("type")
            return v

        def _sort_key(x):
            return x["type"]

        def _collate(ys):
            # we can't collate the count data as a tensor since its dimension
            # will differ between samples. therefore, we return it as a list
            # instead.
            data = [y.pop("data") for y in ys]
            return {"data": data, **default_collate(ys)}

        return {
            k: _collate([_remove_key(v) for v in vs])
            for k, vs in it.groupby(sorted(xs, key=_sort_key), key=_sort_key)
        }

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + get("global_step"))
        np.random.seed(np.random.randint(np.iinfo(np.int32).max) + n)

    return DataLoader(
        dataset, collate_fn=_collate, worker_init_fn=_worker_init, **kwargs
    )
