# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import itertools as it

from functools import partial

import os

import sys

from imageio import imread

import numpy as np

import torch as t

from .logging import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    log,
    set_level,
)
from .analyze import analyze, analyze_genes
from .network import Histonet, STD
from .train import train
from .utility import (
    read_data,
    set_rng_seed,
    store_state,
)


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def _train(
        histonet,
        std,
        optimizer,
        image,
        data,
        label,
        workers=None,
        seed=None,
        **kwargs,
):
    if seed is not None:
        set_rng_seed(seed)

        if workers is None:
            log(WARNING,
                'setting workers to 0 to avoid race conditions '
                '(set --workers explicitly to override)')
            workers = 0

    train(
        histonet,
        std,
        optimizer,
        image,
        label,
        data,
        workers=workers,
        device=DEVICE,
        **kwargs,
    )


def _analyze(
        histonet,
        std,
        image,
        analysis_type,
        data=None,
        label=None,
        **kwargs,
):
    if analysis_type == 'default':
        analyze(histonet, std, image, data, label, **kwargs)
    elif analysis_type == 'genes':
        analyze_genes(histonet, std, image, **kwargs)
    else:
        raise ValueError(f'invalid analysis type "{analysis_type}"')



def main():
    import argparse as ap

    parser = ap.ArgumentParser()

    parser.add_argument('data-dir', type=str)
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    parser.add_argument('--state', type=str)

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(f=_train)

    train_parser.add_argument('--latent-size', type=int, default=16)
    train_parser.add_argument('--zoom', type=float, default=1.)
    train_parser.add_argument('--genes', type=int)
    train_parser.add_argument('--patch-size', type=int, default=700)
    train_parser.add_argument('--batch-size', type=int, default=5)

    train_parser.add_argument('--image-interval', type=int, default=100)
    train_parser.add_argument('--chkpt-interval', type=int, default=100)
    train_parser.add_argument('--workers', type=int)
    train_parser.add_argument('--seed', type=int)
    train_parser.add_argument('--learning-rate', type=float, default=1e-5)

    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.set_defaults(f=_analyze)

    analyze_subparsers = analyze_parser.add_subparsers()

    analyze_default = analyze_subparsers.add_parser('default')
    analyze_default.set_defaults(analysis_type='default')

    analyze_gene = analyze_subparsers.add_parser('genes')
    analyze_gene.set_defaults(analysis_type='genes')
    analyze_gene.add_argument('which', nargs='+', type=str)

    opts = vars(parser.parse_args())

    run_func = opts.pop('f')

    def _go():
        from . import __version__
        log(INFO, '%s %s', __package__, __version__)
        log(DEBUG, 'invocation: %s', ' '.join(sys.argv))
        log(INFO, '%s', ', '.join([f'{k}={v}' for k, v in opts.items()]))

        log(DEBUG, 'using device: %s', str(DEVICE))

        state = opts.pop('state')
        if state is None and run_func is not _train:
            raise RuntimeError('--state must be provided if not training')
        if state is not None:
            state = t.load(state, map_location='cpu')

        data_dir = opts.pop('data-dir')

        image = imread(os.path.join(data_dir, 'image.tif'))

        label_path = os.path.join(data_dir, 'label.tif')
        data_path = os.path.join(data_dir, 'data.csv.gz')
        if all(map(os.path.exists, (label_path, data_path))):
            label = imread(label_path)
            if run_func is _train:
                genes = opts.pop('genes')
            if state is not None:
                genes = state['genes']
            data = read_data(data_path, genes=genes)
        else:
            label = data = None

        if run_func is _train:
            zoom_level = opts.pop('zoom')
        else:
            zoom_level = None
        if state is not None:
            zoom_level = state['zoom_level']
        if zoom_level < 1:
            from scipy.ndimage.interpolation import zoom
            if label is not None:
                label = zoom(label, (zoom_level, zoom_level), order=0)
            if image is not None:
                image = zoom(image, (zoom_level, zoom_level, 1))

        if state is not None:
            histonet = (
                Histonet(**state['model_init_args']['histonet'])
                .to(DEVICE)
            )
            std = (
                STD(**state['model_init_args']['std'])
                .to(DEVICE)
            )
            histonet.load_state_dict(state['model']['histonet'])
            std.load_state_dict(state['model']['std'])
            if run_func is _train:
                opts.pop('latent_size')
        elif run_func is _train:
            histonet = (
                Histonet(latent_size=opts.pop('latent_size'))
                .to(DEVICE)
            )
            std = (
                STD(
                    genes=data.columns,
                    gene_baseline=data.values.mean(0),
                )
                .to(DEVICE)
            )
            t.nn.init.normal_(
                histonet.mixture_loadings[-1].weight,
                std=1e-5,
            )
            t.nn.init.normal_(
                histonet.mixture_loadings[-1].bias,
                mean=-np.log(np.mean(np.bincount(label.flatten())[1:]) * 50),
                std=1e-5,
            )
        else:
            raise RuntimeError()

        image = t.tensor(image).permute(2, 0, 1).float() / 255 * 2 - 1

        if run_func is _train:
            optimizer = t.optim.Adam(
                it.chain(
                    histonet.parameters(),
                    std.parameters(),
                ),
                lr=opts.pop('learning_rate'),
            )
            if state is not None:
                optimizer.load_state_dict(state['optimizer']['default'])

            extra_kwargs = {
                'start_epoch': 1 if state is None else state['epoch'] + 1,
                'store_state': partial(
                    store_state,
                    zoom_level=zoom_level,
                    genes=list(data.columns),
                ),
                'optimizer': optimizer,
            }
        else:
            extra_kwargs = {}

        run_func(
            histonet=histonet,
            std=std,
            image=image,
            data=data,
            label=label,
            **opts,
            **extra_kwargs,
        )

    os.makedirs(opts['output_prefix'], exist_ok=True)

    with open(os.path.join(opts['output_prefix'], 'log'), 'a') as log_file:
        from logging import StreamHandler
        from .logging import Formatter, LOGGER
        log_handler = StreamHandler(log_file)
        log_handler.setFormatter(Formatter(fancy_formatting=False))
        LOGGER.addHandler(log_handler)
        set_level(DEBUG)

        try:
            _go()
        except Exception as err:
            from traceback import format_exc
            from .logging import LOGGER
            trace = err.__traceback__
            while trace.tb_next is not None:
                trace = trace.tb_next
            frame = trace.tb_frame
            LOGGER.findCaller = (
                lambda self, stack_info=None, f=frame:
                (f.f_code.co_filename, f.f_lineno, f.f_code.co_name, None)
            )
            log(ERROR, str(err))
            log(DEBUG, format_exc().strip())
            sys.exit(1)


if __name__ == '__main__':
    main()
