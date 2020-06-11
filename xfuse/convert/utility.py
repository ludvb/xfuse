from typing import Dict, List, NamedTuple, Tuple
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from ..logging import DEBUG, WARNING, log
from ..utility import compute_tissue_mask


class Spot(NamedTuple):
    r"""
    Data type for encoding circular capture spots, which are used in
    technologies based on the Spatial Transcriptomics method
    (https://doi.org/10.1126/science.aaf2403 ).
    """
    x: float
    y: float
    r: float


def rescale(
    image: np.ndarray, scaling_factor: float, resample: int = Image.NEAREST
) -> np.ndarray:
    r"""
    Rescales image

    :param image: Image array
    :param scaling_factor: Scaling factor
    :param resample: Resampling filter
    :returns: The rescaled image
    """
    image = Image.fromarray(image)
    image = image.resize(
        [round(x * scaling_factor) for x in image.size], resample=resample,
    )
    image = np.array(image)
    return image


def labels_from_spots(dst: np.ndarray, spots: List[Spot]) -> None:
    r"""Fills `dst` with labels enumerated from `spots`"""
    for i, s in enumerate(spots, 1):
        x, y, radius = [int(round(x)) for x in (s.x, s.y, s.r)]
        dst[
            tuple(
                zip(
                    *(
                        (y - dy, x - dx)
                        for dy in range(-radius, radius + 1)
                        for dx in range(-radius, radius + 1)
                        if dy ** 2 + dx ** 2 <= s.r ** 2
                    )
                )
            )
        ] = i


def crop_image(
    image: np.ndarray, spots: List[Spot], margin: float = 0.12
) -> np.ndarray:
    r"""Crops `image`, keeping a fixed minimum margin to the `spots`."""
    cs = [[s.x, s.y] for s in spots]

    xmin, ymin = np.min(cs, 0)
    xmax, ymax = np.max(cs, 0)

    margin = margin * max(xmax - xmin, ymax - ymin)

    xmin = int(round(xmin - margin))
    xmax = int(round(xmax + margin))
    ymin = int(round(ymin - margin))
    ymax = int(round(ymax + margin))

    xmin, ymin = [max(a, 0) for a in (xmin, ymin)]

    return image[ymin:ymax, xmin:xmax]


def mask_tissue(
    image: np.ndarray, counts: pd.DataFrame, label: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    r"""
    Detects the tissue in `image`. The area outside of the tissue is given a
    new label with zero counts everywhere.
    """
    mask = compute_tissue_mask(image)

    counts.index += 1
    label[label != 0] += 1

    in_mask = np.unique(label[mask & (label != 0)])
    label[~mask.astype(bool) & ~np.isin(label, in_mask)] = 1

    counts = pd.concat(
        [
            pd.DataFrame(
                [np.repeat(0, counts.shape[1])],
                columns=counts.columns,
                index=[1],
            ).astype(pd.SparseDtype("float", 0)),
            counts,
        ]
    )

    return counts, label


def write_data(
    counts: pd.DataFrame,
    image: np.ndarray,
    label: np.ndarray,
    annotation: Dict[str, np.ndarray],
    type_label: str,
    path: str = "data.h5",
) -> None:
    r"""Writes data to the format used by XFuse."""
    if image.shape[:2] != label.shape[:2]:
        raise RuntimeError(
            f"Image shape ({image.shape[:2]}) is not equal to"
            f" the shape of the label image ({label.shape[:2]})."
        )

    if np.max(image.shape[:2]) > 5000:
        log(
            WARNING,
            "The image resolution is very large! 😱"
            " XFuse typically works best on medium resolution images"
            " (approximately 1000x1000 px)."
            " If you experience performance issues, please consider reducing"
            " the resolution.",
        )

    if counts.columns.duplicated().any():
        log(
            WARNING,
            "Count matrix contains duplicated columns."
            " Counts will be summed by column name.",
        )
        counts = counts.sum(axis=1, level=0)

    log(DEBUG, "writing data to %s", path)
    os.makedirs(os.path.normpath(os.path.dirname(path)), exist_ok=True)
    with h5py.File(path, "w") as data_file:
        data = (
            counts.astype(pd.SparseDtype("float", 0.0)).sparse.to_coo().tocsr()
        )
        data_file.create_dataset(
            "counts/data", data.data.shape, float, data.data.astype(float)
        )
        data_file.create_dataset(
            "counts/indices",
            data.indices.shape,
            data.indices.dtype,
            data.indices,
        )
        data_file.create_dataset(
            "counts/indptr", data.indptr.shape, data.indptr.dtype, data.indptr
        )
        data_file.create_dataset(
            "counts/columns",
            counts.columns.shape,
            h5py.string_dtype(),
            counts.columns.values,
        )
        data_file.create_dataset(
            "counts/index", counts.index.shape, int, counts.index.astype(int)
        )
        data_file.create_dataset("image", image.shape, np.uint8, image)
        data_file.create_dataset("label", label.shape, np.int16, label)
        data_file.create_group("annotation", track_order=True)
        for k, v in annotation.items():
            data_file.create_dataset(f"annotation/{k}", v.shape, np.uint16, v)
        data_file.create_dataset(
            "type", data=type_label, dtype=h5py.string_dtype()
        )
