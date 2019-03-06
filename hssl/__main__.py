# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import os

import sys

from imageio import imread

import torch as t

from .logging import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    log,
    set_level,
)
from .train import train
from .utility import read_data, set_rng_seed


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def _train(
        image,
        data,
        label,
        genes=None,
        zoom=1.,
        workers=None,
        seed=None,
        **kwargs,
):
    if zoom < 1:
        from scipy.ndimage.interpolation import zoom as zoom_fnc
        label = zoom_fnc(label, (zoom, zoom), order=0)
        image = zoom_fnc(image, (zoom, zoom, 1))

    if seed is not None:
        set_rng_seed(seed)

        if workers is None:
            log(WARNING,
                'setting workers to 0 to avoid race conditions '
                '(set --workers explicitly to override)')
            workers = 0

    train(
        image,
        label,
        data,
        workers=workers,
        device=DEVICE,
        **kwargs,
    )


def _analyze(
        image,
        data=None,
        label=None,
):
    raise NotImplementedError()


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

    train_parser.add_argument('--latent-size', type=int, default=100)
    train_parser.add_argument('--zoom', type=float, default=1.)
    train_parser.add_argument('--genes', type=int, default=50)
    train_parser.add_argument('--patch-size', type=int, default=700)
    train_parser.add_argument('--batch-size', type=int, default=5)

    train_parser.add_argument('--image-interval', type=int, default=100)
    train_parser.add_argument('--chkpt-interval', type=int, default=100)
    train_parser.add_argument('--workers', type=int)
    train_parser.add_argument('--seed', type=int)
    train_parser.add_argument('--learning-rate', type=float, default=1e-5)

    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.set_defaults(f=_analyze)

    opts = vars(parser.parse_args())

    def _go():
        log(DEBUG, 'invocation: %s', ' '.join(sys.argv))
        log(INFO, '%s', ', '.join([f'{k}={v}' for k, v in opts.items()]))

        data_dir = opts.pop('data-dir')

        image = imread(os.path.join(data_dir, 'image.tif'))

        label_path = os.path.join(data_dir, 'label.tif')
        data_path = os.path.join(data_dir, 'data.csv.gz')
        if all(map(os.path.exists, (label_path, data_path))):
            label = imread(label_path)
            data = read_data(data_path, genes=opts.pop('genes'))
        else:
            label = data = None

        log(DEBUG, 'using device: %s', str(DEVICE))

        opts.pop('f')(image, data, label, **opts)

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
