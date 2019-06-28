from functools import partial

from contextlib import ExitStack

from copy import deepcopy

import numpy as np

import pyro as p

import torch as t
from torch.utils.tensorboard.writer import SummaryWriter

from .handlers.stats import (
    FactorActivationHistogram,
    FactorActivationMaps,
    FactorActivationMean,
    FactorActivationSummary,
    Image,
    Latent,
    LogLikelihood,
    RMSE,
)
from .logging import DEBUG, set_level
from .model import XFuse
from .model.experiment import ST


def __remove_this():
    import os
    import pandas as pd
    import pyvips
    from .utility import design_matrix_from, read_data
    from .dataset import Dataset, RandomSlide, collate

    design_file = os.path.expanduser(
        # '~/histonet-test-data/mob-0.1-validation/design.small.csv')
        '~/histonet-test-data/hdst-mob/design.csv')

    design = pd.read_csv(design_file)
    design_dir = os.path.dirname(design_file)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(design_dir, p)
        )

    count_data = read_data(map(_path, design.data))

    design_matrix = design_matrix_from(design[[
        x for x in design.columns
        if x not in [
                'name',
                'image',
                'labels',
                'validation',
                'data',
        ]
    ]])

    dataset = Dataset(
        [
            RandomSlide(
                data=counts,
                image=pyvips.Image.new_from_file(_path(image)),
                label=pyvips.Image.new_from_file(_path(labels)),
                patch_size=224,
            )
            for image, labels, counts in zip(
                design.image,
                design.labels,
                (count_data.loc[x] for x in count_data.index.levels[0]),
            )
        ],
        design_matrix,
    )

    loader = t.utils.data.DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    genes = list(count_data.columns)

    return (
        dataset, loader, genes,
        count_data.mean().mean() / spot_size(dataset),
        t.as_tensor(count_data.mean(0).values).log(),
        count_data,
    )


def dim_red(x, mask=None, method='pca', n_components=3, **kwargs):
    if method != 'pca':
        raise NotImplementedError()

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=bool)
    elif isinstance(mask, t.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)

    from sklearn.decomposition import PCA

    if isinstance(x, t.Tensor):
        x = x.detach().cpu().numpy()

    values = (
        PCA(n_components=n_components, **kwargs)
        .fit_transform(x[mask])
    )

    dst = np.zeros((*mask.shape, n_components))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst


def prep(x):
    if isinstance(x, t.Tensor):
        return x.to(t.device('cuda'))
    if isinstance(x, list):
        return [prep(y) for y in x]
    if isinstance(x, dict):
        return {k: prep(v) for k, v in x.items()}


from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.path.join('/tmp/tb/', datetime.now().isoformat()))

set_level(DEBUG)

from .dataset import spot_size
data, loader, genes, scale, baseline, counts = __remove_this()
xfuse = XFuse([
    ST(
        n=len(data),
        default_scale=scale,
        factors=[
            (0., baseline),
            (-10, None),
        ],
    )
]).to(t.device('cuda'))
import pyro.optim
svi = p.infer.SVI(
    xfuse.model,
    xfuse.guide,
    p.optim.Adam({'lr': 1e-3}),
    p.infer.Trace_ELBO(),
)
fixed_x = prep(next(iter(loader)))


def normalize(img):
    mins = img.permute(0, 2, 3, 1).reshape(-1, img.shape[1]).min(0).values[None, :, None, None]
    maxs = img.permute(0, 2, 3, 1).reshape(-1, img.shape[1]).max(0).values[None, :, None, None]
    return (img - mins) / (maxs - mins)


global_step = 0

def do():
    global global_step
    stats_trackers = [
        (1, LogLikelihood),
        (1, RMSE),
        (1, partial(
            FactorActivationMean,
            list(xfuse._get_experiment('ST').factors.keys()),
        )),
        (10, partial(
            FactorActivationHistogram,
            list(xfuse._get_experiment('ST').factors.keys()),
        )),
        (100, partial(
            FactorActivationMaps,
            list(xfuse._get_experiment('ST').factors.keys()),
        )),
        (100, FactorActivationSummary),
        (100, Image),
        (100, Latent),
    ]
    for x in loader:
        with ExitStack() as stack:
            for tracker in [
                    tracker for (freq, tracker) in stats_trackers
                    if global_step % freq == 0
            ]:
                stack.enter_context(tracker(writer, global_step))
            loss = svi.step(prep(x))
            writer.add_scalar('loss/elbo', loss, global_step)
            print(f'{np.mean(loss)}')
            global_step += 1

    print('starting factor purge')
    with t.no_grad():
        def _model_without(n):
            reduced_model = deepcopy(xfuse)
            reduced_model._XFuse__experiment_store['ST'].remove_factor(n)
            return reduced_model

        reduced_models, ns = zip(*[
            (_model_without(n), n)
            for n in xfuse._get_experiment('ST').factors
        ])

        def _compare_once():
            guide = p.poutine.trace(xfuse.guide).get_trace(fixed_x)

            def _evaluate(model):
                return (
                    p.poutine.trace(p.poutine.replay(model, guide))
                    .get_trace(fixed_x)
                    .log_prob_sum()
                    .item()
                )

            full = _evaluate(xfuse.model)
            deltas = [
                _evaluate(xfuse.model) - full for xfuse in reduced_models]
            return deltas

        res = [_compare_once() for _ in range(10)]
        res = np.array(res).mean(0)
        dubious = [
            n for res, n in reversed(sorted(zip(res, ns)))
            if res >= 0
        ]
        if dubious == []:
            print('no factors are dubious')
            xfuse._get_experiment('ST').add_factor((-10., None))
        else:
            print(
                'the following factors are dubious: '
                + ', '.join(map(str, dubious))
            )
            for n in dubious[:-1][:len(res) - 2]:
                xfuse._get_experiment('ST').remove_factor(n)
