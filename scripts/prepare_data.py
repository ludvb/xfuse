#!/usr/bin/env python3

# pylint: skip-file

import os
from collections import namedtuple

import numpy as np
import pandas as pd
from imageio import imread
from scipy.ndimage.interpolation import zoom
from tifffile import imwrite
from tqdm import tqdm

import tissue_recognition as tr

Spot = namedtuple("Spot", ["x", "y", "r"])


def run(image, counts, spots):
    label = np.zeros(image.shape[:2]).astype(np.int16)

    print("filling labels")
    iterator = tqdm(spots, dynamic_ncols=True)
    for i, s in enumerate(iterator, 1):
        x, y, r = [int(round(x)) for x in (s.x, s.y, s.r)]
        label[
            tuple(
                zip(
                    *(
                        (y - dy, x - dx)
                        for dy in range(-r, r + 1)
                        for dx in range(-r, r + 1)
                        if dy ** 2 + dx ** 2 <= s.r ** 2
                    )
                )
            )
        ] = i

    cs = [[s.x, s.y] for s in spots]

    xmin, ymin = np.min(cs, 0)
    xmax, ymax = np.max(cs, 0)

    xmin -= 0.2 * (xmax - xmin)
    xmax += 0.2 * (xmax - xmin)
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    xmin, xmax, ymin, ymax = [int(round(x)) for x in (xmin, xmax, ymin, ymax)]
    xmin, ymin = [max(a, 0) for a in (xmin, ymin)]

    image = image[ymin:ymax, xmin:xmax]
    label = label[ymin:ymax, xmin:xmax]

    # add count values for sites outside the tissue
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    print("running tissue recognition...")
    tr.recognize_tissue(image.copy(), mask)
    mask = tr.get_binary_mask(mask)

    counts.n += 1
    label[label != 0] += 1

    counts = pd.concat(
        [
            pd.DataFrame(
                [[1, *np.repeat(0, counts.shape[1] - 1)]],
                columns=counts.columns,
            ).astype(pd.SparseDtype("float", 0)),
            counts,
        ]
    )
    if counts.columns.duplicated().any():
        counts = counts.sum(axis=1, level=0)
    label[np.invert(mask).astype(bool)] = 1

    return counts, image, label


def main():
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--counts")
    parser.add_argument("--spots")
    parser.add_argument("--output-directory")
    parser.add_argument("--compress", default=0)
    parser.add_argument("--zoom", type=float)
    parser.add_argument(
        "--validation",
        type=float,
        help="proportion of spots to hold out for validation",
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=100,
        help="filter spots with a low total number of reads",
    )

    opts = vars(parser.parse_args())

    validation_prop = opts.pop("validation")
    assert validation_prop is None or 0 < validation_prop < 1

    min_counts = opts.pop("min_counts")

    output_directory = opts.pop("output_directory")
    os.makedirs(output_directory)

    compression = opts.pop("compress")

    image = imread(opts.pop("image"))

    zoom_level = opts.pop("zoom")
    if zoom_level is not None:
        image = zoom(image, (zoom_level, zoom_level, 1.0), order=0)

    counts = pd.read_csv(opts.pop("counts"), sep="\t", index_col=0)
    counts = counts[counts.sum(1) >= min_counts]
    spots = opts.pop("spots")
    if spots is not None:
        spots = pd.read_csv(spots, sep="\t")
        if zoom_level is not None:
            spots[["pixel_x", "pixel_y"]] *= zoom_level
        counts = counts.loc[
            spots[["x", "y"]].apply(lambda x: "x".join(map(str, x)), 1)
        ]
        xmax, xmin = [f(spots.x) for f in (np.max, np.min)]
        pxmax, pxmin = [
            np.mean(spots.pixel_x[spots.x == x]) for x in (xmax, xmin)
        ]
        r = (pxmax - pxmin) / (xmax - xmin) / 4
        spots = list(
            spots[["pixel_x", "pixel_y"]].apply(lambda x: Spot(*x, r), 1)
        )
    else:
        r = np.sqrt(np.product(image.shape[:2]) / 32 / 34) / 4
        spots = [
            Spot(
                *[
                    round((float(y) - 1) / d * s)
                    for y, s, d in zip(
                        x.split("x"), image.shape[:2][::-1], (32, 34)
                    )
                ],
                r,
            )
            for x in counts.index
        ]

    counts.insert(0, "n", [*range(1, len(counts) + 1)])

    counts, image, label = run(image, counts, spots)

    print("writing counts...")
    with h5py.file(os.path.join(opts.output_directory, "data.h5"), "w") as f:
        counts_reindexed = counts.set_index("n")
        data = counts_reindexed.sparse.to_coo().tocsr()
        f.create_dataset(
            "matrix/data", data.data.shape, float, data.data.astype(float)
        )
        f.create_dataset(
            "matrix/indices",
            data.indices.shape,
            data.indices.dtype,
            data.indices,
        )
        f.create_dataset(
            "matrix/indptr", data.indptr.shape, data.indptr.dtype, data.indptr
        )
        f.create_dataset(
            "matrix/columns",
            counts_reindexed.columns.shape,
            h5py.string_dtype(),
            counts_reindexed.columns.values,
        )
        f.create_dataset(
            "matrix/index",
            counts_reindexed.index.shape,
            int,
            counts_reindexed.index.values.astype(int),
        )

    print("writing image...")
    imwrite(
        os.path.join(output_directory, "image.tif"),
        image,
        tile=(256, 256),
        compress=compression,
    )

    spot_labels = (
        counts.reset_index()
        .rename(index=str, columns={"index": "spot"})[["spot", "n"]]
        .set_index("n")
        .iloc[1:]
    )

    if validation_prop:
        validation_spots = np.random.choice(
            [*filter(lambda x: x > 1, counts.n)],
            int(validation_prop * (len(counts.n) - 2)),
            replace=False,
        )
        validation = label.copy()
        validation[np.invert(np.isin(label, [0, 1, *validation_spots]))] = 0
        label[np.isin(label, validation_spots)] = 0

        spot_labels["validation"] = 0
        spot_labels.loc[validation_spots, "validation"] = 1

        print("writing validation labels...")
        imwrite(
            os.path.join(output_directory, "validation.tif"),
            validation,
            tile=(256, 256),
            compress=compression,
        )

    print("writing labels...")
    imwrite(
        os.path.join(output_directory, "label.tif"),
        label,
        tile=(256, 256),
        compress=compression,
    )
    spot_labels.to_csv(os.path.join(output_directory, "spot_labels.csv.gz"))


if __name__ == "__main__":
    main()
