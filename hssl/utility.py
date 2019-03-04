import itertools as it

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch as t

from torchvision.utils import make_grid

from .logging import INFO, log


def set_rng_seed(seed: int):
    import random
    i32max = np.iinfo(np.int32).max
    random.seed(seed)
    n_seed = random.choice(range(i32max + 1))
    t_seed = random.choice(range(i32max + 1))
    np.random.seed(n_seed)
    t.manual_seed(t_seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    log(INFO, ' / '.join([
        'random rng seeded with %d',
        'numpy rng seeded with %d',
        'torch rng seeded with %d',
    ]), seed, n_seed, t_seed)


def store_state(model, optimizers, epoch, file):
    t.save(
        dict(
            model=model.state_dict(),
            optimizers=[x.state_dict() for x in optimizers],
            epoch=epoch,
        ),
        file,
    )


def restore_state(model, optimizers, file):
    state = t.load(file)
    model.load_state_dict(state['model'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    return state['epoch']


def zip_dicts(ds):
    d0 = next(ds)
    d = {k: [] for k in d0.keys()}
    for d_ in it.chain([d0], ds):
        for k, v in d_.items():
            try:
                d[k].append(v)
            except AttributeError:
                raise ValueError('dict keys are inconsistent')
    return d


def collect_items(d):
    d_ = {}
    for k, v in d.items():
        try:
            d_[k] = v.item()
        except (ValueError, AttributeError):
            pass
    return d_


def center_crop(input, target_shape):
    return input[tuple([
        slice((a - b) // 2, (a - b) // 2 + b)
        if b is not None else
        slice(None)
        for a, b in zip(input.shape, target_shape)
    ])]


def visualize_batch(batch, normalize=False, **kwargs):
    return plt.imshow(
        np.transpose(
            make_grid(
                batch.detach().cpu(),
                nrow=int(np.floor(np.sqrt(len(batch)))),
                padding=5,
                normalize=normalize,
            ),
            (1, 2, 0),
        ),
        **kwargs,
    )


def read_data(path, filter_ambiguous=True, genes=None):
    data = pd.read_csv(path).set_index('n')
    if filter_ambiguous:
        data = data[[
            x for x in data.columns if 'ambiguous' not in x
        ]]
    if genes:
        data = data[
            data.sum(0)
            .sort_values()
            [-genes:]
            .index
        ]
    return data
