from typing import Dict, Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix

from .utility import (
    Spot,
    crop_image,
    labels_from_spots,
    mask_tissue,
    rescale,
    write_data,
)


def run(
    image: np.ndarray,
    bc_matrix: h5py.File,
    tissue_positions: pd.DataFrame,
    spot_radius: float,
    output_file: str,
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
    mask: bool = True,
) -> None:
    r"""
    Converts data from the 10X SpaceRanger pipeline for visium arrays into
    the data format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    counts = csr_matrix(
        (
            bc_matrix["matrix"]["data"],
            bc_matrix["matrix"]["indices"],
            bc_matrix["matrix"]["indptr"],
        ),
        shape=(
            bc_matrix["matrix"]["barcodes"].shape[0],
            bc_matrix["matrix"]["features"]["name"].shape[0],
        ),
    )
    counts = pd.DataFrame.sparse.from_spmatrix(
        counts.astype(float),
        columns=bc_matrix["matrix"]["features"]["name"][()].astype(str),
        index=pd.Index([*range(1, counts.shape[0] + 1)], name="n"),
    )

    if scale_factor is not None:
        tissue_positions[["x", "y"]] *= scale_factor
        spot_radius *= scale_factor
        image = rescale(image, scale_factor, Image.BICUBIC)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }

    spots = list(
        tissue_positions.loc[
            bc_matrix["matrix"]["barcodes"][()].astype(str)
        ].apply(lambda x: Spot(x=x["x"], y=x["y"], r=spot_radius), 1)
    )

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
