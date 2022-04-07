from typing import Dict, Optional

import cv2 as cv
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse import csr_matrix

from ..utility.core import rescale
from .utility import (
    Spot,
    find_margin,
    labels_from_spots,
    mask_tissue,
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
    custom_mask: Optional[np.ndarray] = None,
    rotate: bool = False,
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
        image = rescale(image, scale_factor, Image.BOX)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }
        if custom_mask is not None:
            custom_mask = rescale(custom_mask, scale_factor, Image.NEAREST)

    spots = list(
        tissue_positions[["x", "y"]]
        .loc[bc_matrix["matrix"]["barcodes"][()].astype(str)]
        .apply(lambda x: Spot(x=x["x"], y=x["y"], r=spot_radius), 1)
    )

    label = np.zeros(image.shape[:2]).astype(np.int16)
    labels_from_spots(label, spots)

    col_mask, row_mask = find_margin(image)
    image = image[row_mask][:, col_mask]
    label = label[row_mask][:, col_mask]
    if custom_mask is not None:
        custom_mask = custom_mask[row_mask][:, col_mask]

    if scale_factor is not None:
        # The outermost pixels may belong in part to the margin if we
        # downscaled the image. Therefore, remove one extra row/column.
        image = image[1:-1, 1:-1]
        label = label[1:-1, 1:-1]
        if custom_mask is not None:
            custom_mask = custom_mask[1:-1, 1:-1]

    if mask:
        if custom_mask is not None:
            initial_mask = custom_mask
        else:
            (in_tissue_idxs,) = np.where(
                tissue_positions["in_tissue"]
                .loc[bc_matrix["matrix"]["barcodes"][()].astype(str)]
                .values
            )
            in_tissue_idxs = in_tissue_idxs + 1
            in_tissue = np.where(np.isin(label, in_tissue_idxs), True, False)
            idx1, idx2 = distance_transform_edt(
                label == 0, return_indices=True, return_distances=False
            )
            initial_mask = np.where(
                label != 0,
                np.where(in_tissue, cv.GC_FGD, cv.GC_BGD),
                np.where(in_tissue[idx1, idx2], cv.GC_PR_FGD, cv.GC_PR_BGD),
            )
            initial_mask = initial_mask.astype(np.uint8)
        counts, label = mask_tissue(
            image, counts, label, initial_mask=initial_mask
        )

    write_data(
        counts,
        image,
        label,
        type_label="ST",
        annotation={
            k: (v, {x: str(x) for x in np.unique(v)})
            for k, v in annotation.items()
        },
        auto_rotate=rotate,
        path=output_file,
    )
