from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom

from .utility import (
    Spot,
    crop_image,
    labels_from_spots,
    mask_tissue,
    write_data,
)


def run(
    counts: pd.DataFrame,
    image: np.ndarray,
    output_file: str,
    spots: Optional[pd.DataFrame] = None,
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
) -> None:
    r"""
    Converts data from the Spatial Transcriptomics pipeline into the data
    format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    if scale_factor is not None:
        image = zoom(image, (scale_factor, scale_factor, 1.0), order=0)
        annotation = {
            k: zoom(v, (scale_factor, scale_factor), order=0)
            for k, v in annotation.items()
        }
        if spots is not None:
            spots[["pixel_x", "pixel_y"]] *= scale_factor

    if spots is not None:
        counts = counts.loc[
            spots[["x", "y"]].apply(lambda x: "x".join(map(str, x)), 1)
        ]
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
        radius = np.sqrt(np.product(image.shape[:2]) / 32 / 34) / 4
        spots = [
            Spot(  # type: ignore
                *[
                    round((float(y) - 1) / d * s)
                    for y, s, d in zip(
                        x.split("x"), image.shape[:2][::-1], (32, 34)
                    )
                ],
                radius,
            )
            for x in counts.index
        ]

    counts.index = pd.Index([*range(1, counts.shape[0] + 1)], name="n")

    label = np.zeros(image.shape[:2]).astype(np.int16)
    labels_from_spots(label, spots)

    image = crop_image(image, spots)
    label = crop_image(label, spots)
    annotation = {k: crop_image(v, spots) for k, v in annotation.items()}

    counts, label = mask_tissue(image, counts, label)

    write_data(
        counts,
        image,
        label,
        type_label="ST",
        annotation=annotation,
        path=output_file,
    )
