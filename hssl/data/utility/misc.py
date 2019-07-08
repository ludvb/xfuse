import itertools as it

import numpy as np

import torch as t
from torch.utils.data.dataloader import default_collate

from ..dataset import Dataset
from ...session import get_global_step


__all__ = [
    'make_dataloader',
    'spot_size',
]


def spot_size(dataset: Dataset):
    return np.median(np.concatenate([
        np.bincount(d['label'].flatten())
        for d in dataset
    ]))


def make_dataloader(*args, **kwargs):
    def _collate(xs):
        def _remove_key(v):
            v.pop('type')
            return v

        def _sort_key(x):
            return x['type']

        def _collate(ys):
            # we can't collate the count data as a tensor since its dimension
            # will differ between samples. therefore, we return it as a list
            # instead.
            data = [y.pop('data') for y in ys]
            return {
                'data': data,
                **default_collate(ys),
            }

        return {
            k: _collate([_remove_key(v) for v in vs])
            for k, vs in it.groupby(sorted(xs, key=_sort_key), key=_sort_key)
        }

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + get_global_step())
        np.random.seed(np.random.randint(
            np.iinfo(np.int32).max) + n)

    return t.utils.data.DataLoader(
        *args,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        **kwargs,
    )
