import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .utility import (
    Spot,
    crop_image,
    labels_from_spots,
    mask_tissue,
    rescale,
    write_data,
)


def run(
    counts: pd.DataFrame,
    image: np.ndarray,
    output_file: str,
    spots: Optional[pd.DataFrame] = None,
    transformation: Optional[np.ndarray] = None,
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
    mask: bool = True,
) -> None:
    r"""
    Converts data from the Spatial Transcriptomics pipeline into the data
    format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    if scale_factor is not None:
        image = rescale(image, scale_factor, Image.BICUBIC)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }
        if spots is not None:
            spots[["pixel_x", "pixel_y"]] *= scale_factor
        if transformation is not None:
            scale_matrix = np.array(
                [[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]]
            )
            transformation = transformation @ scale_matrix

    if spots is not None:
        spots.index = spots[["x", "y"]].apply(
            lambda x: "x".join(map(str, x)), 1
        )
        spot_names = np.intersect1d(spots.index, counts.index)
        spots = spots.loc[spot_names]
        counts = counts.loc[spot_names]
        xmax, xmin = [f(spots.x) for f in (np.max, np.min)]
        pxmax, pxmin = [
            np.mean(spots.pixel_x[spots.x == x]) for x in (xmax, xmin)
        ]
        radius = (pxmax - pxmin) / (xmax - xmin) / 4
        spots = list(
            spots[["pixel_x", "pixel_y"]].apply(
                lambda x: Spot(*x, radius),  # type: ignore
                1,
            )
        )
    else:
        warnings.warn(
            "Converting data from the Spatial Transcriptomics pipeline"
            " without a spot detector file has been deprecated and will be"
            " removed in a future version.",
            DeprecationWarning,
        )
        coordinates = np.array(
            [
                [float(x), float(y)]
                for x, y in (x.split("x") for x in counts.index)
            ]
        )
        if transformation is not None:
            coordinates = np.concatenate(
                [coordinates, np.ones((len(coordinates), 1))], axis=-1
            )
            coordinates = coordinates @ transformation
            coordinates = coordinates[:, :2]
        else:
            coordinates[:, 0] = (coordinates[:, 0] - 1) / 32 * image.shape[1]
            coordinates[:, 1] = (coordinates[:, 1] - 1) / 34 * image.shape[0]
        radius = np.sqrt(np.product(image.shape[:2]) / 32 / 34) / 4
        spots = [Spot(x=x, y=y, r=radius) for x, y in coordinates]

    counts.index = pd.Index([*range(1, counts.shape[0] + 1)], name="n")

    label = np.zeros(image.shape[:2]).astype(np.int16)
    labels_from_spots(label, spots)

    image = crop_image(image, spots)
    label = crop_image(label, spots)
    annotation = {k: crop_image(v, spots) for k, v in annotation.items()}

    if mask:
        counts, label = mask_tissue(image, counts, label)

    write_data(
        counts,
        image,
        label,
        type_label="ST",
        annotation=annotation,
        path=output_file,
    )
