from functools import wraps

import itertools as it

import signal

from typing import List, Optional, Set, Tuple

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


def compose(f, *gs):
    return (
        (lambda *args, **kwargs: f(compose(*gs)(*args, **kwargs)))
        if gs != ()
        else f
    )


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


def read_data(
        paths: List[str],
        filter_ambiguous: bool = True,
        num_genes: int = None,
        genes: List[str] = None,
) -> pd.DataFrame:
    def _load_file(p):
        log(INFO, 'loading data file %s', p)
        return pd.read_csv(p, index_col=0, dtype=float)

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

    if genes is not None:
        data_ = pd.DataFrame(
            np.zeros((len(data), len(genes))),
            columns=genes,
            index=data.index,
            dtype=float,
        )
        shared_genes = np.intersect1d(genes, data.columns)
        data_[shared_genes] = data[shared_genes]
        data = data_
    elif filter_ambiguous:
        data = data[[
            x for x in data.columns if 'ambiguous' not in x
        ]]

    if num_genes:
        if isinstance(num_genes, int):
            data = data[
                data.sum(0)
                .sort_values()
                [-num_genes:]
                .index
            ]
        if isinstance(num_genes, list):
            data = data[num_genes]

    return data


def design_matrix_from(
        design: pd.DataFrame,
        covariates: Optional[List[Tuple[str, Set[str]]]] = None,
) -> pd.DataFrame:
    if len(design.columns) == 0:
        return pd.DataFrame(np.zeros((0, len(design))))

    design = (
        design
        [[x for x in sorted(design.columns)]]
        .astype(str)
        .astype('category')
    )

    if covariates is not None:
        missing_covariates = [
            x for x, _ in covariates if x not in design.columns]
        if missing_covariates != []:
            raise ValueError(
                'the following covariates are missing from the design: '
                + ', '.join(missing_covariates)
            )

        for covariate, values in covariates:
            design[covariate].cat.set_categories(sorted(values), inplace=True)
        design = design[[x for x, _ in covariates]]
    else:
        for covariate in design.columns:
            design[covariate].cat.set_categories(
                sorted(design[covariate].cat.categories), inplace=True)

    def _encode(covariate):
        log(INFO, 'encoding design covariate "%s" with %d categories: %s',
            covariate.name, len(covariate.cat.categories),
            ', '.join(covariate.cat.categories))
        return pd.DataFrame(
            (
                np.eye(len(covariate.cat.categories), dtype=int)
                [:, covariate.cat.codes]
            ),
            index=covariate.cat.categories,
        )

    ks, vs = zip(*[(k, _encode(v)) for k, v in design.iteritems()])
    return pd.concat(vs, keys=ks)


def argmax(x: t.Tensor):
    return np.unravel_index(t.argmax(x), x.shape)


def integrate_loadings(
        loadings: t.Tensor,
        label: t.Tensor,
        max_label: int = None,
):
    if max_label is None:
        max_label = t.max(label)
    return (
        t.einsum(
            'btxy,bxyi->it',
            loadings.exp(),
            (
                t.eye(max_label + 1)
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


def chunks_of(n, xs):
    class FillMarker:
        pass
    return map(
        lambda xs: [*filter(lambda x: x is not FillMarker, xs)],
        it.zip_longest(*[iter(xs)]*n, fillvalue=FillMarker),
    )
