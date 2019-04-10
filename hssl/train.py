from collections import namedtuple

import gzip

import itertools as it

import os

from typing import List, Optional

from imageio import imwrite

import numpy as np

import torch as t

from .analyze import dim_red, visualize_batch
from .dataset import Dataset, collate
from .logging import INFO, log
from .utility import (
    chunks_of,
    collect_items,
    integrate_loadings,
    with_interrupt_handler,
    zip_dicts,
)
from .utility.state import State, save_state, to_device


class Interrupted(Exception):
    pass


class Step(t.nn.Module):
    def __init__(self, histonet, std, dataset_size):
        super().__init__()
        self.histonet = histonet
        self.std = std
        self.m = dataset_size

    def forward(self, x):
        try:
            # FIXME: see .dataset:collate
            effects = x['effects']
        except KeyError:
            effects = None

        _z, img_distr, loadings, _state = self.histonet(x['image'])

        lpimg = img_distr.log_prob(x['image'])

        integrated_loadings = integrate_loadings(loadings, x['label'])
        d = self.std(integrated_loadings[1:], effects)

        lpobs = d.log_prob(x['data'])

        hdkl = self.histonet.complexity_cost(len(x) / self.m)
        sdkl = self.std.complexity_cost(len(x) / self.m)
        dkl = hdkl + sdkl
        img_loss = -t.sum(lpimg)
        xpr_loss = -t.sum(lpobs)

        loss = img_loss + xpr_loss + dkl

        return {k: v.unsqueeze(0) for k, v in {
            'L': loss,
            'Li': img_loss,
            'Lx': xpr_loss,
            'dqp': dkl,
            'rmse': t.mean(
                t.sqrt(t.mean((d.mean - x['data']) ** 2, 1))
                .masked_select(x['data'].sum(1) != 0)
            ),
        }.items()}


def train(
        state: State,
        output_prefix: str,
        dataset: Dataset,
        dataset_validation: Optional[Dataset] = None,
        batch_size: int = 5,
        image_interval: Optional[int] = 50,
        chkpt_interval: Optional[int] = 10000,
        valid_interval: Optional[int] = 10,
        epochs: Optional[int] = None,
        workers: int = None,
        devices: Optional[List[t.device]] = None,
) -> State:
    if workers is None:
        workers = len(os.sched_getaffinity(0))

    if devices is None:
        devices = [*map(t.device, (
            [f'cuda:{i}' for i in range(t.cuda.device_count())]
            if t.cuda.device_count() > 0 else
            ['cpu']
        ))]

    state = to_device(state, devices[0])

    histonet = state.histonet
    std = state.std
    optimizer = state.optimizer
    epoch = state.epoch + 1

    stepper = Step(histonet, std, len(dataset))

    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    def _worker_init(n):
        np.random.seed(np.random.get_state()[1][0] + epoch)
        np.random.seed(np.random.randint(
            np.iinfo(np.int32).max) + n)

    training_data = t.utils.data.DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=_worker_init,
        shuffle=True,
    )
    validation_data = (
        t.utils.data.DataLoader(
            dataset_validation,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=workers,
            worker_init_fn=_worker_init,
            shuffle=False,
        )
        if
        dataset_validation is not None
        and valid_interval is not None
        else
        None
    )

    fixed_data = next(iter(training_data))

    def _step(xs):
        inputs = [
            {k: v.to(dev) for k, v in dat.items()}
            for dev, dat in zip(devices, xs)
        ]
        outputs = t.nn.parallel.gather(
            t.nn.parallel.parallel_apply(
                t.nn.parallel.replicate(stepper, devices[:len(inputs)]),
                inputs,
            ),
            devices[0],
        )

        if stepper.training:
            optimizer.zero_grad()
            t.sum(outputs['L']).backward()
            optimizer.step()

        return collect_items({k: t.mean(v) for k, v in outputs.items()})

    t.enable_grad()
    histonet.train()

    def _report(epoch, output, dataset_size, validation):
        iteration = output.pop('iteration')

        log(
            INFO,
            ' '.join([
                'epoch {epoch:6d} of {epochs:s}'.format(
                    epoch=epoch,
                    epochs=str(epochs) if epochs is not None else '∞',
                ),
                ' '
                f'(%0{len(str(dataset_size))}d/{dataset_size})'
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
                            str(int(validation)),
                            str(k),
                            str(v),
                        ])
                        + '\n'
                    ).encode())

    def _save_image(epoch):
        z, img_distr, loadings, state = histonet(
            fixed_data['image'].to(devices[0]))

        for data, prefix in [
                ((fixed_data['image'] + 1) / 2, 'he'),
                (
                    (
                        (
                            img_distr
                            .sample()
                            .clamp(-1, 1)
                            + 1
                        )
                        / 2
                    ),
                    'he-sample',
                ),
                ((img_distr.mean.clamp(-1, 1) + 1) / 2, 'he-mean'),
                (dim_red(z.permute(0, 2, 3, 1)).transpose(0, 3, 1, 2), 'z'),
                (dim_red(loadings.permute(0, 2, 3, 1)).transpose(0, 3, 1, 2),
                 'fct'),
                (
                    dim_red(
                        (
                            (
                                loadings.exp()
                                / loadings.exp().sum(1).unsqueeze(1)
                            )
                            .permute(0, 2, 3, 1)
                        ),
                    ).transpose(0, 3, 1, 2),
                    'fct-rel',
                ),
                (dim_red(state.permute(0, 2, 3, 1)).transpose(0, 3, 1, 2),
                 'state'),
        ]:
            imwrite(
                os.path.join(img_prefix, f'{prefix}-epoch-{epoch:06d}.png'),
                visualize_batch(data),
            )

        t.enable_grad()
        histonet.train()

    def _interrupt_handler(*_):
        from multiprocessing import current_process
        if current_process().name == 'MainProcess':
            log(INFO, 'interrupted')
            raise Interrupted()

    @with_interrupt_handler(_interrupt_handler)
    def _run_training():
        nonlocal epoch

        for epoch in it.takewhile(
                lambda x: epochs is None or x <= epochs,
                it.count(epoch),
        ):
            def _step_with(data: t.utils.data.DataLoader):
                for output in map(
                        lambda x: zip_dicts(
                            map(
                                lambda y: {'iteration': y[0], **y[1]},
                                filter(lambda y: y is not None, x),
                            ),
                        ),
                        chunks_of(100, (
                            (i, _step(xs)) for i, xs in enumerate(
                                chunks_of(len(devices), data),
                                1,
                            )
                        ))
                ):
                    _report(
                        epoch,
                        output,
                        int(np.ceil(len(data) / len(devices))),
                        not stepper.training,
                    )

            stepper.train()
            t.enable_grad()

            _step_with(training_data)

            stepper.eval()
            t.no_grad()

            if validation_data is not None and epoch % valid_interval == 0:
                _step_with(validation_data)
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
