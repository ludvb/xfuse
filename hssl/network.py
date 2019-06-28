from functools import partial

from contextlib import ExitStack

from copy import deepcopy

from datetime import datetime

import os

import numpy as np

import pyro as p
from pyro.optim import Adam

import torch as t
from torch.utils.tensorboard.writer import SummaryWriter

from .dataset import spot_size
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
from .utility import to_device


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


writer = SummaryWriter(os.path.join('/tmp/tb/', datetime.now().isoformat()))

set_level(DEBUG)

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
svi = p.infer.SVI(
    xfuse.model,
    xfuse.guide,
    Adam({'lr': 1e-3}),
    p.infer.Trace_ELBO(),
)
global_step = 0


def do(epoch):
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
            loss = svi.step(to_device(x, t.device('cuda')))
            writer.add_scalar('loss/elbo', loss, global_step)
            print(f'{loss}')
            global_step += 1

    if epoch % 50 == 0:
        print('starting factor purge')

        def _model_without(n):
            reduced_model = deepcopy(xfuse)
            reduced_model._XFuse__experiment_store['ST'].remove_factor(n)
            return reduced_model

        reduced_models, ns = zip(*[
            (_model_without(n), n)
            for n in xfuse._get_experiment('ST').factors
        ])

        def _compare_on(x):
            def _once():
                guide = p.poutine.trace(xfuse.guide).get_trace(x)

                def _evaluate(model):
                    with p.poutine.trace() as tr, \
                         p.poutine.block(
                             hide_fn=lambda x: not x['is_observed']), \
                         p.poutine.replay(trace=guide):
                        model(x)
                    return tr.trace.log_prob_sum().item()

                full = _evaluate(xfuse.model)
                deltas = [_evaluate(xfuse.model) - full
                          for xfuse in reduced_models]
                return deltas
            return np.mean([_once() for _ in range(5)], 0)

        with t.no_grad():
            res = [_compare_on(to_device(x, t.device('cuda')))
                   for x, _ in zip(loader, range(5))]
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
