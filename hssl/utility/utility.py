from functools import partial, wraps

import itertools as it

import signal

import numpy as np

import pandas as pd

import torch as t

from ..logging import INFO, log


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


def read_data(paths, filter_ambiguous=True, genes=None):
    def _load_file(p):
        log(INFO, 'loading data file %s', p)
        return pd.read_csv(p, index_col=0)

    ks, vs = zip(*[(p, _load_file(p)) for p in paths])
    data = (
        pd.concat(
            vs,
            keys=ks,
            join='outer',
            axis=0,
            sort=False,
        )
        .fillna(0)
    )

    data = data.iloc[:, (data.sum(0) > 0).values]

    if filter_ambiguous:
        data = data[[
            x for x in data.columns if 'ambiguous' not in x
        ]]

    if genes:
        if isinstance(genes, int):
            data = data[
                data.sum(0)
                .sort_values()
                [-genes:]
                .index
            ]
        if isinstance(genes, list):
            data = data[genes]

    return data


def design_matrix_from(design: pd.DataFrame) -> pd.DataFrame:
    if len(design.columns) == 0:
        return pd.DataFrame(np.zeros((0, len(design))))

    design = design.astype(str).astype('category')

    def _encode(factor):
        log(INFO, 'encoding design factor "%s" with %d categories: %s',
            factor.name, len(factor.cat.categories),
            ', '.join(factor.cat.categories))
        return pd.DataFrame(
            (
                np.eye(len(factor.cat.categories), dtype=int)
                [:, factor.cat.codes]
            ),
            index=factor.cat.categories,
        )

    ks, vs = zip(*[(k, _encode(v)) for k, v in design.iteritems()])
    return pd.concat(vs, keys=ks)


def argmax(x: t.Tensor):
    return np.unravel_index(t.argmax(x), x.shape)


def integrate_loadings(loadings: t.Tensor, label: t.Tensor):
    return (
        t.einsum(
            'btxy,bxyi->it',
            t.exp(loadings),
            (
                t.eye(t.max(label) + 1)
                .to(label)
                [label.flatten()]
                .reshape(*label.shape, -1)
                .float()
            ),
        )
    )


def with_interrupt_handler(handler):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            previous_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handler)
            func(*args, **kwargs)
            signal.signal(signal.SIGINT, previous_handler)
        return _wrapper
    return _decorator


def lazify(computation):
    def _g():
        result = computation()
        while True:
            yield result
    g = _g()

    def _wrapped():
        return next(g)
    return _wrapped


def chunks_of(n, xs):
    class FillMarker:
        pass
    return map(
        lambda xs: [*filter(lambda x: x is not FillMarker, xs)],
        it.zip_longest(*[iter(xs)]*n, fillvalue=FillMarker),
    )
