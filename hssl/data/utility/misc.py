import itertools as it
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate  # type: ignore

from ...session import get
from ...utility import center_crop
from ..dataset import Dataset

__all__ = ["make_dataloader", "spot_size"]


def spot_size(dataset: Dataset) -> np.float64:
    r"""Computes the median spot size in the :class:`Dataset`"""
    return np.median(
        np.concatenate(
            [
                np.bincount(d["label"].flatten())[1:]
                for d in it.islice(dataset, 1)  # type: ignore
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

            # Crop image sizes to the minimum size over the batch
            min_size = {}
            for y in ys:
                for k, v in y.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    if k in min_size:
                        min_size[k] = torch.min(
                            min_size[k], torch.as_tensor(v.shape)
                        )
                    else:
                        min_size[k] = torch.as_tensor(v.shape)
            for y in ys:
                for k, v in min_size.items():
                    if k in y and isinstance(y[k], torch.Tensor):
                        y[k] = center_crop(y[k], v.numpy().tolist())

            return {"data": data, **default_collate(ys)}

        return {
            k: _collate([_remove_key(v) for v in vs])
            for k, vs in it.groupby(sorted(xs, key=_sort_key), key=_sort_key)
        }

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + get("training_data").step)
        np.random.seed(np.random.randint(np.iinfo(np.int32).max) + n)

    return DataLoader(
        dataset, collate_fn=_collate, worker_init_fn=_worker_init, **kwargs
    )
