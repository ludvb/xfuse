import itertools as it

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_fill_holes

from .core import rescale, resize
from ..logging import INFO, log


def remove_fg_elements(mask: np.ndarray, size_threshold: float):
    r"""Removes small foreground elements"""
    labels, _ = label(mask)
    labels_unique, label_counts = np.unique(labels, return_counts=True)
    small_labels = labels_unique[
        label_counts < size_threshold ** 2 * np.prod(mask.shape)
    ]
    mask[np.isin(labels, small_labels)] = False
    return mask


def compute_tissue_mask(
    image: np.ndarray,
    convergence_threshold: float = 0.0001,
    size_threshold: float = 0.01,
) -> np.ndarray:
    r"""
    Computes boolean mask indicating likely foreground elements in histology
    image.
    """
    # pylint: disable=no-member
    # ^ pylint fails to identify cv.* members
    original_shape = image.shape[:2]
    scale_factor = 1000 / max(original_shape)

    image = rescale(image, scale_factor, resample=Image.NEAREST)
    initial_mask = binary_fill_holes(
        cv.blur(cv.Canny(cv.blur(image, (5, 5)), 100, 200), (5, 5))
    )

    mask = np.where(initial_mask, cv.GC_PR_FGD, cv.GC_PR_BGD)
    mask = mask.astype(np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = bgd_model.copy()

    log(INFO, "Computing tissue mask:")

    for i in it.count(1):
        old_mask = mask.copy()
        cv.grabCut(
            image, mask, None, bgd_model, fgd_model, 1, cv.GC_INIT_WITH_MASK,
        )
        prop_changed = (mask != old_mask).sum() / np.prod(mask.shape)
        log(INFO, f"  Iteration {i}: {prop_changed=}")
        if prop_changed < convergence_threshold:
            break

    mask = mask == cv.GC_PR_FGD
    mask = cleanup_mask(mask, size_threshold)

    mask = resize(mask, target_shape=original_shape, resample=Image.NEAREST)

    return mask


def cleanup_mask(mask: np.ndarray, size_threshold: float):
    r"""Removes small background and foreground elements"""
    mask = ~remove_fg_elements(~mask, size_threshold)
    mask = remove_fg_elements(mask, size_threshold)
    return mask
