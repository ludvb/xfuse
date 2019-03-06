import gzip

import itertools as it

import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import torch as t

from .analyze import dim_red, visualize_batch
from .dataset import Dataset
from .logging import INFO, log
from .network import Histonet
from .utility import (
    collect_items,
    store_state as _store_state,
    zip_dicts,
)


def train(
        histonet: Histonet,
        optimizer: t.optim.Optimizer,
        image: np.ndarray,
        label: np.ndarray,
        data: pd.DataFrame,
        output_prefix: str,
        patch_size: int = 700,
        batch_size: int = 5,
        image_interval: int = 50,
        chkpt_interval: int = 10000,
        workers: int = None,
        device: t.device = None,
        start_epoch: int = 1,
        store_state=None,
):
    if workers is None:
        workers = len(os.sched_getaffinity(0))

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if store_state is None:
        store_state = _store_state

    epoch = start_epoch

    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    dataset = Dataset(image, label, data, patch_size=patch_size)

    def _collate(xs):
        data = [x.pop('data') for x in xs]
        nlabels = [len(d) - 1 for d in data]
        labels = [
            (l + n) * (l != 0).long() for l, n in
            zip((x.pop('label') for x in xs), it.accumulate((0, *nlabels)))
        ]
        return dict(
            data=t.cat([data[0], *(d[1:] for d in data[1:])]),
            label=t.stack(labels),
            **{k: t.stack([x[k] for x in xs]) for k in xs[0].keys()},
        )

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + epoch)
        np.random.seed(np.random.randint(
            np.iinfo(np.int32).max) + n)

    dataloader = t.utils.data.DataLoader(
        dataset,
        collate_fn=_collate,
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=_worker_init,
    )

    fixed_data = next(iter(dataloader))
    fixed_data['label'].zero_()

    def _run_histonet_on(x):
        spatial_data = (
            x['data']
            [x['label'].flatten()]
            .reshape(*x['label'].shape, -1)
            .permute(0, 3, 1, 2)
        )

        return histonet(t.cat([x['image'], spatial_data], 1))

    def _step(x):
        x = {k: v.to(device) for k, v in x.items()}

        z, img_mu, img_sd, rate, logit, _state = _run_histonet_on(x)

        lpimg = (
            t.distributions.Normal(img_mu, img_sd)
            .log_prob(x['image'])
        )

        label = (
            t.eye(t.max(x['label']) + 1, device=device)
            [x['label'].flatten()]
            .reshape(*x['label'].shape, -1)
            .float()
        )
        rates = t.einsum('bgxy,bxyi->ig', rate, label)

        d = t.distributions.NegativeBinomial(
            rates[1:] + 1e-10,
            logits=logit.t(),
        )

        data = x['data'][1:, 1:]
        lpobs = d.log_prob(data)

        dkl = histonet.complexity_cost(len(x) / len(dataset))

        img_loss = -t.sum(lpimg)
        xpr_loss = -t.sum(lpobs)
        loss = img_loss + xpr_loss + dkl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return collect_items({
            'L': loss,
            'Li': img_loss,
            'Lx': xpr_loss,
            'dqp': dkl,
            'p(img|z)': t.mean(t.exp(lpimg)),
            'p(xpr|z)': t.mean(t.exp(lpobs)),
            'rmse': t.mean(t.sqrt(t.mean((d.mean - data) ** 2, 1))),
        })

    t.enable_grad()
    histonet.train()

    def _report(epoch, output):
        iteration = output.pop('iteration')

        log(
            INFO,
            ' '.join([
                f'epoch {epoch:5d}',
                ' '
                f'(%0{len(str(len(dataloader)))}d/{len(dataloader)})'
                % iteration[-1],
                ' :: ',
                '  //  '.join([
                    '{} = {:>9s}'.format(k, f'{np.mean(v):.2e}')
                    for k, v in output.items()
                ]),
            ])
        )

        with gzip.open(
                os.path.join(output_prefix, 'training_data.csv.gz'),
                'ab',
        ) as fh:
            for k, vs in output.items():
                for i, v in zip(iteration, vs):
                    fh.write((
                        ','.join([
                            str(epoch),
                            str(i),
                            str(k),
                            str(v),
                        ])
                        + '\n'
                    ).encode())

    def _save_chkpt(epoch):
        store_state(
            histonet,
            optimizer,
            epoch,
            os.path.join(
                chkpt_prefix,
                f'epoch-{epoch:05d}.pkl',
            ),
        )

    def _save_image(epoch):
        t.no_grad()
        histonet.eval()

        z, mu, sd, rate, logit, state = _run_histonet_on({
            k: v.to(device) for k, v in fixed_data.items()
        })

        xpr = rate * t.exp(logit.t()[..., None, None])

        for data, prefix in [
                (
                    (
                        t.distributions.Normal(mu, sd)
                        .sample()
                        .clamp(0, 1)
                    ),
                    'he',
                ),
                (dim_red(z), 'z'),
                (dim_red(xpr), 'xpr'),
                (dim_red(xpr / xpr.sum(1).unsqueeze(1)), 'xpr-rel'),
                (dim_red(state), 'state'),
        ]:
            visualize_batch(data)
            plt.savefig(
                os.path.join(img_prefix, f'{prefix}-epoch-{epoch:06d}.png'),
                dpi=200,
            )

        t.enable_grad()
        histonet.train()

    for epoch in it.count(epoch):
        for output in map(
                lambda x: zip_dicts(
                    map(
                        lambda y: {'iteration': y[0], **y[1]},
                        filter(lambda y: y is not None, x),
                    ),
                ),
                it.zip_longest(
                    *[(
                        (i, _step(x)) for i, x in enumerate(dataloader, 1)
                    )] * 10
                ),
        ):
            _report(epoch, output)

        if epoch % image_interval == 0:
            _save_image(epoch)
        if epoch % chkpt_interval == 0:
            _save_chkpt(epoch)
