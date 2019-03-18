#!/usr/bin/env python3

from collections import namedtuple

import itertools as it

import os

from imageio import imread

import numpy as np

import pandas as pd

from scipy.ndimage.interpolation import zoom

from tifffile import imwrite

from tqdm import tqdm


Spot = namedtuple('Spot', ['x', 'y', 'r'])


def run(image, counts, spots):
    label = np.zeros(image.shape[:2]).astype(np.int16)

    print('filling labels')
    iterator = tqdm(spots, dynamic_ncols=True)
    for i, s in enumerate(iterator, 1):
        x, y, r = [int(round(x)) for x in (s.x, s.y, s.r)]
        label[
            tuple(zip(*(
                (y - dy, x - dx)
                for dy, dx in
                filter(
                    lambda x: np.sum(np.array(x) ** 2) <= s.r ** 2,
                    it.product(
                        range(-r, r + 1),
                        range(-r, r + 1),
                    )
                )
            )))
        ] = i

    cs = [[s.x, s.y] for s in spots]

    xmin, ymin = np.min(cs, 0)
    xmax, ymax = np.max(cs, 0)

    xmin -= 0.1 * (xmax - xmin)
    xmax += 0.1 * (xmax - xmin)
    ymin -= 0.1 * (ymax - ymin)
    ymax += 0.1 * (ymax - ymin)

    xmin, xmax, ymin, ymax = [
        int(round(x)) for x in (xmin, xmax, ymin, ymax)]
    xmin, ymin = [max(a, 0) for a in (xmin, ymin)]

    return (
        counts,
        image[ymin:ymax, xmin:xmax],
        label[ymin:ymax, xmin:xmax],
    )


def main():
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--counts')
    parser.add_argument('--spots')
    parser.add_argument('--output-directory')
    parser.add_argument('--compress', default=0)
    parser.add_argument('--zoom', type=float)

    opts = vars(parser.parse_args())

    output_directory = opts.pop('output_directory')
    os.makedirs(output_directory)

    compression = opts.pop('compress')

    image = imread(opts.pop('image'))

    zoom_level = opts.pop('zoom')
    if zoom_level is not None:
        image = zoom(image, (zoom_level, zoom_level, 1.), order=0)

    counts = pd.read_csv(opts.pop('counts'), sep='\t', index_col=0)
    spots = opts.pop('spots')
    if spots is not None:
        spots = pd.read_csv(spots, sep='\t')
        if zoom_level is not None:
            spots[['pixel_x', 'pixel_y']] *= zoom_level
        counts = counts.loc[
            spots[['x', 'y']]
            .apply(lambda x: 'x'.join(map(str, x)), 1)
        ]
        xmax, xmin = [f(spots.x) for f in (np.max, np.min)]
        pxmax, pxmin = [
            np.mean(spots.pixel_x[spots.x == x]) for x in (xmax, xmin)]
        r = (pxmax - pxmin) / (xmax - xmin) / 4
        spots = list(
            spots[['pixel_x', 'pixel_y']]
            .apply(lambda x: Spot(*x, r), 1)
        )
    else:
        r = np.sqrt(np.product(image.shape[:2]) / 32 / 34) / 4
        spots = [
            Spot(
                *[
                    round((float(y) - 1) / d * s)
                    for y, s, d in zip(
                        x.split('x'), image.shape[:2][::-1], (32, 34))
                ],
                r,
            )
            for x in counts.index
        ]

    counts = counts.reset_index().rename({'index': 'n'}, axis='columns')
    counts['n'] = [*range(1, len(counts) + 1)]

    counts, image, label = run(image, counts, spots)

    print('writing counts...')
    counts.to_csv(
        os.path.join(output_directory, 'data.csv.gz'),
        sep=',',
        index=False,
    )
    print('writing image...')
    imwrite(
        os.path.join(output_directory, 'image.tif'),
        image,
        tile=(256, 256),
        compress=compression,
    )
    print('writing labels...')
    imwrite(
        os.path.join(output_directory, 'label.tif'),
        label,
        tile=(256, 256),
        compress=compression,
    )


if __name__ == '__main__':
    main()
