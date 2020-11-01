import itertools as it
from typing import Optional

import cv2 as cv
import numpy as np
from scipy.ndimage import label

from .core import center_crop
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
    initial_mask: Optional[np.ndarray] = None,
    convergence_threshold: float = 0.0001,
    size_threshold: float = 0.01,
) -> np.ndarray:
    r"""
    Computes boolean mask indicating likely foreground elements in histology
    image.
    """
    # pylint: disable=no-member
    # ^ pylint fails to identify cv.* members
    if initial_mask is None:
        initial_mask = np.zeros(image.shape[:2], dtype=np.bool)
        initial_mask_center = center_crop(
            initial_mask,
            tuple(int(round(x * 0.8)) for x in iter(initial_mask.shape)),
        )
        initial_mask_center[...] = True

    mask = cv.GC_PR_BGD * np.ones(image.shape[:2], dtype=np.uint8)
    mask[initial_mask] = cv.GC_PR_FGD

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

    return mask


def cleanup_mask(mask: np.ndarray, size_threshold: float):
    r"""Removes small background and foreground elements"""
    mask = ~remove_fg_elements(~mask, size_threshold)
    mask = remove_fg_elements(mask, size_threshold)
    return mask
