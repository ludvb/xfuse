import gzip

import itertools as it

import os

from typing import Optional

from imageio import imwrite

import numpy as np

import torch as t

from .analyze import dim_red, visualize_batch
from .dataset import Dataset, collate
from .logging import INFO, log
from .utility import (
    collect_items,
    integrate_loadings,
    with_interrupt_handler,
    zip_dicts,
)
from .utility.state import State, save_state


class Interrupted(Exception):
    pass


def train(
        state: State,
        dataset: Dataset,
        output_prefix: str,
        batch_size: int = 5,
        image_interval: Optional[int] = 50,
        chkpt_interval: Optional[int] = 10000,
        epochs: Optional[int] = None,
        workers: int = None,
        device: t.device = None,
) -> State:
    if workers is None:
        workers = len(os.sched_getaffinity(0))

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    histonet = state.histonet
    std = state.std
    optimizer = state.optimizer
    epoch = state.epoch + 1

    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + epoch)
        np.random.seed(np.random.randint(
            np.iinfo(np.int32).max) + n)

    dataloader = t.utils.data.DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=_worker_init,
        shuffle=True,
    )

    fixed_data = next(iter(dataloader))

    def _step(x):
        x = {k: v.to(device) for k, v in x.items()}

        z, img_mu, img_sd, loadings, _state = histonet(x['image'])

        lpimg = (
            t.distributions.Normal(img_mu, img_sd)
            .log_prob(x['image'])
        )

        integrated_loadings = integrate_loadings(loadings, x['label'])
        rate, logit = std(integrated_loadings[1:], x['effects'])
        d = t.distributions.NegativeBinomial(rate, logits=logit)

        data = x['data']
        lpobs = d.log_prob(data)

        hdkl = histonet.complexity_cost(len(x) / len(dataset))
        sdkl = std.complexity_cost(len(x) / len(dataset))
        dkl = hdkl + sdkl

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
                'epoch {epoch:6d} of {epochs:s}'.format(
                    epoch=epoch,
                    epochs=str(epochs) if epochs is not None else '∞',
                ),
                ' '
                f'(%0{len(str(len(dataloader)))}d/{len(dataloader)})'
                % iteration[-1],
                ' :: ',
                '  |  '.join([
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

    def _save_image(epoch):
        t.no_grad()
        histonet.eval()

        z, mu, sd, loadings, state = histonet(fixed_data['image'].to(device))

        for data, prefix in [
                (
                    (
                        (
                            t.distributions.Normal(mu, sd)
                            .sample()
                            .clamp(-1, 1)
                            + 1
                        )
                        / 2
                    ),
                    'he',
                ),
                (dim_red(z), 'z'),
                (dim_red(loadings), 'fct'),
                (dim_red(loadings.exp() / loadings.exp().sum(1).unsqueeze(1)),
                 'fct-rel'),
                (dim_red(state), 'state'),
        ]:
            imwrite(
                os.path.join(img_prefix, f'{prefix}-epoch-{epoch:06d}.png'),
                visualize_batch(data),
            )

        t.enable_grad()
        histonet.train()

    def _interrupt_handler(*_):
        log(INFO, 'interrupted')
        raise Interrupted()

    @with_interrupt_handler(_interrupt_handler)
    def _run_training():
        nonlocal epoch

        for epoch in it.takewhile(
                lambda x: epochs is None or x <= epochs,
                it.count(epoch),
        ):
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

            if image_interval is not None and epoch % image_interval == 0:
                _save_image(epoch)
            if chkpt_interval is not None and epoch % chkpt_interval == 0:
                save_state(
                    State(
                        histonet=histonet,
                        std=std,
                        optimizer=optimizer,
                        epoch=epoch,
                    ),
                    os.path.join(
                        chkpt_prefix,
                        f'epoch-{epoch:06d}.pkl',
                    ),
                )

    try:
        _run_training()
    except Interrupted:
        pass

    return State(
        histonet=histonet,
        std=std,
        optimizer=optimizer,
        epoch=epoch,
    )
