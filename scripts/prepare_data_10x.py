#!/usr/bin/env python3

# pylint: skip-file

import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from imageio import imread
from scipy.ndimage.interpolation import zoom
from scipy.sparse import csr_matrix
from tifffile import imwrite

from prepare_data import Spot
from prepare_data import run as _run

SCALE = 0.1039393
R = 143.31714641818704 / 2


def run(
    image_file: str,
    counts_file: str,
    spots_file: str,
    zoom_factor: Optional[float] = None,
    min_counts: Optional[float] = 1,
):
    image = imread(image_file)
    if zoom_factor is not None:
        image = zoom(image, (zoom_factor, zoom_factor, 1.0), order=0)
    else:
        zoom_factor = 1.0

    with h5py.File(counts_file, "r") as data:
        counts = csr_matrix(
            (
                data["matrix"]["data"],
                data["matrix"]["indices"],
                data["matrix"]["indptr"],
            )
        )
        counts = pd.DataFrame.sparse.from_spmatrix(
            counts.astype(float),
            columns=data["matrix"]["features"]["name"][()].astype(str),
            index=data["matrix"]["barcodes"][()].astype(str),
        )
        counts = counts.iloc[:, np.array(counts.sum(0)).flatten() > 0]
        mask = np.array([True] * counts.shape[0])
        if min_counts is not None:
            mask &= counts.sum(1) >= min_counts
        counts = counts[mask]
        counts.insert(0, "n", [*range(1, len(counts) + 1)])

        spots = pd.read_csv(spots_file, index_col=0, header=None)[
            [4, 5]
        ].rename(columns={4: "y", 5: "x"})
        spots = spots.loc[data["matrix"]["barcodes"][()].astype(str)]
        spots = spots[mask]
        spots = list(
            spots.apply(
                lambda x: Spot(
                    x=x["x"] * SCALE * zoom_factor,
                    y=x["y"] * SCALE * zoom_factor,
                    r=R * SCALE * zoom_factor,
                ),
                1,
            )
        )

    return _run(image, counts, spots)


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

    opts = parser.parse_args()

    counts, image, label = run(
        opts.image, opts.counts, opts.spots, opts.zoom, opts.min_counts
    )

    os.makedirs(opts.output_directory, exist_ok=True)

    print("writing counts...")
    with h5py.File(os.path.join(opts.output_directory, "data.h5"), "w") as f:
        counts_reindexed = counts.set_index("n")
        data = (
            counts_reindexed.astype(pd.SparseDtype("float", 0.0))
            .sparse.to_coo()
            .tocsr()
        )
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
        os.path.join(opts.output_directory, "image.tif"),
        image,
        tile=(256, 256),
        compress=opts.compress,
    )

    spot_labels = (
        counts.reset_index()
        .rename(index=str, columns={"index": "spot"})[["spot", "n"]]
        .set_index("n")
        .iloc[1:]
    )

    if opts.validation:
        validation_spots = np.random.choice(
            [*filter(lambda x: x > 1, counts.n)],
            int(opts.validation * (len(counts.n) - 2)),
            replace=False,
        )
        validation = label.copy()
        validation[np.invert(np.isin(label, [0, 1, *validation_spots]))] = 0
        label[np.isin(label, validation_spots)] = 0

        spot_labels["validation"] = 0
        spot_labels.loc[validation_spots, "validation"] = 1

        print("writing validation labels...")
        imwrite(
            os.path.join(opts.output_directory, "validation.tif"),
            validation,
            tile=(256, 256),
            compress=opts.compress,
        )

    print("writing labels...")
    imwrite(
        os.path.join(opts.output_directory, "label.tif"),
        label,
        tile=(256, 256),
        compress=opts.compress,
    )
    spot_labels.to_csv(
        os.path.join(opts.output_directory, "spot_labels.csv.gz")
    )


if __name__ == "__main__":
    main()
