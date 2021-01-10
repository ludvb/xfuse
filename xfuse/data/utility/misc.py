import itertools as it
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate  # type: ignore

from ...session import get
from ...utility import center_crop
from ..dataset import Dataset

__all__ = ["make_dataloader", "estimate_spot_size"]


class _RepeatSampler:
    """Sampler that repeats forever."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.DataLoader):
    r"""
    DataLoader that avoids spawning new workers in each epoch.
    See https://github.com/pytorch/pytorch/issues/15849
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(
            self, "batch_sampler", _RepeatSampler(self.batch_sampler)
        )
        self.reset_workers()

    def reset_workers(self):
        r"""
        Reloads worker processes on the next call to `self.__iter__`. This
        should be called if the dataset in the main process has been changed.
        """
        self.__iterator = super().__iter__()
        return self

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            # pylint: disable=stop-iteration-return
            yield next(self.__iterator)


def estimate_spot_size(dataset: Dataset) -> Dict[str, float]:
    r"""Computes the mean spot size in the :class:`Dataset`"""

    def _compute_size(x):
        if x["data_type"] == "ST":
            zero_count_idxs = 1 + torch.where(x["data"].sum(1) == 0)[0]
            partial_idxs = np.unique(
                torch.cat(
                    [
                        x["label"][0],
                        x["label"][-1],
                        x["label"][:, 0],
                        x["label"][:, -1],
                    ]
                )
                .cpu()
                .numpy()
            )
            partial_idxs = np.setdiff1d(
                partial_idxs, zero_count_idxs.cpu().numpy()
            )
            mask = np.invert(
                np.isin(x["label"].cpu().numpy(), [0, *partial_idxs])
            )
            _, sizes = np.unique(
                x["label"].cpu().numpy()[mask].flatten(), return_counts=True,
            )
            return sizes
        raise NotImplementedError()

    return {
        k: np.concatenate([v[1] for v in vs]).mean()
        for k, vs in it.groupby(
            [(x["data_type"], _compute_size(x)) for x in dataset],
            key=lambda x: x[0],
        )
    }


def make_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader:
    r"""Creates a :class:`~torch.utils.data.DataLoader` for `dataset`"""

    def _collate(xs):
        def _remove_key(v):
            v.pop("data_type")
            return v

        def _sort_key(x):
            return x["data_type"]

        def _collate(ys):
            collated_data = {}

            # we can't collate the count data as a tensor since its dimension
            # will differ between samples. therefore, we return it as a list
            # instead.
            try:
                collated_data.update({"data": [y.pop("data") for y in ys]})
            except KeyError:
                pass

            # Collate any other non-tensor as list
            collated_data.update(
                {
                    k: [y.pop(k) for y in ys]
                    for k in set(
                        k
                        for y in ys
                        for k, v in y.items()
                        if not torch.is_tensor(v)
                    )
                }
            )

            # Crop image sizes to the minimum size over the batch
            min_size = {}
            for y in ys:
                for k, v in y.items():
                    if k in min_size:
                        min_size[k] = torch.min(
                            min_size[k], torch.as_tensor(v.shape)
                        )
                    else:
                        min_size[k] = torch.as_tensor(v.shape)
            for y in ys:
                for k, v in min_size.items():
                    y[k] = center_crop(y[k], v.numpy().tolist())
            collated_data.update(default_collate(ys))

            return collated_data

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
