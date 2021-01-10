import os
import warnings
from typing import Dict, List, NamedTuple, Tuple

import h5py
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.ndimage.morphology import binary_fill_holes

from ..logging import INFO, log
from ..utility.mask import compute_tissue_mask, remove_fg_elements
from ..utility.visualization import _normalize


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


def find_min_bbox(
    mask: np.ndarray, rotate: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    r""""Finds the mininum bounding box enclosing a given image mask"""
    contour, _ = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if rotate:
        return cv.minAreaRect(np.concatenate(contour))
    x, y, w, h = cv.boundingRect(np.concatenate(contour))
    return ((x + w // 2, y + h // 2), (w, h), 0.0)


def crop_to_rect(
    img: np.ndarray,
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    interpolation_method: int = cv.INTER_NEAREST,
    margin: float = 0.12,
) -> np.ndarray:
    r""""Crops image to rectangle"""
    width, height = rect[1]
    px_margin = margin * max(width, height)
    width = int(np.round(width + 2 * px_margin))
    height = int(np.round(height + 2 * px_margin))
    rect = (rect[0], (width, height), rect[2])
    box_src = cv.boxPoints(rect)
    box_dst = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype=np.float32,
    )
    transform = cv.getPerspectiveTransform(box_src, box_dst)
    return cv.warpPerspective(
        img,
        transform,
        (width, height),
        flags=interpolation_method,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=np.median(
            np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), 0
        ),
    )


def relabel(
    counts: pd.DataFrame, label: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    >>> counts = pd.DataFrame(index=[1,4,3])
    >>> label = np.array([[1, 0], [4, 4]])
    >>> relabel(counts, label)
    (Empty DataFrame
    Columns: []
    Index: [1, 2], array([[1, 0],
           [2, 2]]))
    """
    idxs = np.unique(label[label != 0])
    counts = counts.loc[idxs]
    counts = counts.rename({old: new for new, old in enumerate(idxs, 1)})
    label = np.searchsorted([0, *idxs], label)
    return counts, label


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


def trim_margin(
    image: np.ndarray, label: np.ndarray, margin_color=None, tol: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Trims margins from `image` and removes the corresponding regions in
    `label`.

    >>> image = np.array([[1, 1], [1, 0]])
    >>> label = np.array([[1, 2], [3, 4]])
    >>> trim_margin(image, label)
    (array([[0]]), array([[4]]))
    """
    if margin_color is None:
        margin_color = image.max((0, 1))

    img_dims = tuple(np.arange(len(image.shape) - len(margin_color.shape)))
    color_dims = tuple(
        np.arange(len(image.shape) - len(margin_color.shape), len(image.shape))
    )

    color_scale = np.max(image, img_dims) - np.min(image, img_dims)

    is_margin = np.ones(image.shape[:2], dtype=bool)
    is_margin &= (image > (margin_color - tol * color_scale)).all(color_dims)
    is_margin &= (image < (margin_color + tol * color_scale)).all(color_dims)

    col_mask = binary_fill_holes(np.invert(is_margin.all(0)))
    row_mask = binary_fill_holes(np.invert(is_margin.all(1)))

    image = image[row_mask][:, col_mask]
    label = label[row_mask][:, col_mask]

    return image, label


def write_data(
    counts: pd.DataFrame,
    image: np.ndarray,
    label: np.ndarray,
    annotation: Dict[str, Tuple[np.ndarray, Dict[int, str]]],
    type_label: str,
    auto_rotate: bool = False,
    path: str = "data.h5",
) -> None:
    r"""Writes data to the format used by XFuse."""
    if image.shape[:2] != label.shape[:2]:
        raise RuntimeError(
            f"Image shape ({image.shape[:2]}) is not equal to"
            f" the shape of the label image ({label.shape[:2]})."
        )

    if np.max(image.shape[:2]) > 5000:
        warnings.warn(
            "The image resolution is very large! 😱"
            " XFuse typically works best on medium resolution images"
            " (approximately 1000x1000 px)."
            " If you experience performance issues, please consider reducing"
            " the resolution."
        )

    if counts.columns.duplicated().any():
        warnings.warn(
            "Count matrix contains duplicated columns."
            " Counts will be summed by column name."
        )
        try:
            counts = counts.sparse.to_dense()
        except AttributeError:
            pass
        # ^ HACK: line below fails with KeyError on pandas 1.1.4 when `counts`
        #         is sparse
        counts = counts.sum(axis=1, level=0)

    data_mask = ~np.isin(label, counts.index[counts.sum(1) == 0])
    data_mask = remove_fg_elements(data_mask, 0.1)
    if not np.all(data_mask == 0):
        rect = find_min_bbox(data_mask, rotate=auto_rotate)
        image = crop_to_rect(image, rect, interpolation_method=cv.INTER_LINEAR)
        label = crop_to_rect(
            label, rect, interpolation_method=cv.INTER_NEAREST
        )
        annotation = {
            k: (
                crop_to_rect(
                    annotation_label,
                    rect,
                    interpolation_method=cv.INTER_NEAREST,
                ),
                label_names,
            )
            for k, (annotation_label, label_names) in annotation.items()
        }

    counts, label = relabel(counts, label)

    image = _normalize(image.astype(np.float32), axis=(0, 1)) * 2 - 1
    image = 0.9 * image
    # ^ reduce contrast to alleviate tension in the extremes due to the tanh
    #   activation in the image decoder

    log(INFO, "Writing data to %s", path)
    os.makedirs(os.path.normpath(os.path.dirname(path)), exist_ok=True)
    with h5py.File(path, "w") as data_file:
        data = csr_matrix(counts.values.astype(float))
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
        data_file.create_dataset("image", image.shape, np.float32, image)
        data_file.create_dataset("label", label.shape, np.int16, label)
        data_file.create_group("annotation", track_order=True)
        for k, (annotation_label, label_names) in annotation.items():
            data_file.create_dataset(
                f"annotation/{k}/label",
                annotation_label.shape,
                np.uint16,
                annotation_label,
            )
            data_file.create_dataset(
                f"annotation/{k}/names/keys",
                len(label_names),
                np.int64,
                list(label_names.keys()),
            )
            data_file.create_dataset(
                f"annotation/{k}/names/values",
                len(label_names),
                h5py.string_dtype(),
                list(label_names.values()),
            )
        data_file.create_dataset(
            "type", data=type_label, dtype=h5py.string_dtype()
        )
