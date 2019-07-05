import itertools as it

import numpy as np

import torch as t
from torch.utils.data.dataloader import default_collate

from ..dataset import Dataset


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
    return t.utils.data.DataLoader(*args, collate_fn=_collate, **kwargs)
