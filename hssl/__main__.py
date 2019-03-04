# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import os

import sys

from imageio import imread

import torch as t

from .logging import DEBUG, INFO, WARNING, log, set_level
from .train import train
from .utility import read_data, set_rng_seed


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def main():
    import argparse as ap

    args = ap.ArgumentParser()

    args.add_argument('data-dir', type=str)

    args.add_argument('--latent-size', type=int, default=100)

    args.add_argument('--zoom', type=float, default=1.)
    args.add_argument('--genes', type=int, default=50)
    args.add_argument('--patch-size', type=int, default=700)
    args.add_argument('--batch-size', type=int, default=5)

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--image-interval', type=int, default=100)
    args.add_argument('--chkpt-interval', type=int, default=100)
    args.add_argument('--workers', type=int)
    args.add_argument('--seed', type=int)
    args.add_argument('--learning-rate', type=float, default=1e-5)

    opts = vars(args.parse_args())

    set_level(DEBUG)
    log(DEBUG, 'invocation: %s', ' '.join(sys.argv))
    log(INFO, '%s', ', '.join([f'{k}={v}' for k, v in opts.items()]))

    data_dir = opts.pop('data-dir')

    image = imread(os.path.join(data_dir, 'image.tif'))
    label = imread(os.path.join(data_dir, 'label.tif'))
    data = read_data(
        os.path.join(data_dir, 'data.csv.gz'),
        genes=opts.pop('genes'),
    )

    zoom_level = opts.pop('zoom')
    if zoom_level < 1:
        from scipy.ndimage.interpolation import zoom
        label = zoom(label, (zoom_level, zoom_level), order=0)
        image = zoom(image, (zoom_level, zoom_level, 1))

    workers = opts.pop('workers')

    seed = opts.pop('seed')
    if seed is not None:
        set_rng_seed(seed)

        if workers is None:
            log(WARNING,
                'setting workers to 0 to avoid race conditions '
                '(set --workers explicitly to override)')
            workers = 0

    log(DEBUG, 'using device: %s', str(DEVICE))

    train(image, label, data, workers=workers, device=DEVICE, **opts)


if __name__ == '__main__':
    main()
