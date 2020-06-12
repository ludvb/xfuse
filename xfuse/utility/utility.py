import itertools as it
from functools import wraps
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..logging import DEBUG, WARNING, log
from ..session import get

__all__ = [
    "checkpoint",
    "center_crop",
    "compute_tissue_mask",
    "cleanup_mask",
    "design_matrix_from",
    "find_device",
    "remove_fg_elements",
    "sparseonehot",
    "isoftplus",
    "to_device",
    "with_",
]


def checkpoint(function, *args, **kwargs):
    r"""
    Wrapper for :func:`torch.utils.checkpoint.checkpoint` that conditions
    checkpointing on the session `eval` state.
    """
    if get("eval"):
        return function(*args, **kwargs)
    return _checkpoint(function, *args, **kwargs)


def center_crop(x, target_shape):
    r"""Crops `x` to the given `target_shape` from the center"""
    return x[
        tuple(
            [
                slice(round((a - b) / 2), round((a - b) / 2) + b)
                if b is not None
                else slice(None)
                for a, b in zip(x.shape, target_shape)
            ]
        )
    ]


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
            [int(round(x * 0.8)) for x in iter(initial_mask.shape)],
        )
        initial_mask_center[...] = True

    mask = cv.GC_PR_BGD * np.ones(image.shape[:2], dtype=np.uint8)
    mask[initial_mask] = cv.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = bgd_model.copy()

    log(DEBUG, "Computing tissue mask:")

    for i in it.count(1):
        old_mask = mask.copy()
        cv.grabCut(
            image, mask, None, bgd_model, fgd_model, 1, cv.GC_INIT_WITH_MASK,
        )
        prop_changed = (mask != old_mask).sum() / np.prod(mask.shape)
        log(DEBUG, f"  Iteration {i}: {prop_changed=}")
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


def design_matrix_from(
    design: Dict[str, Dict[str, Union[int, str]]],
    covariates: Optional[List[Tuple[str, List[str]]]] = None,
) -> pd.DataFrame:
    r"""
    Constructs the design matrix from the design specified in the design file
    """
    design_table = pd.concat(
        [
            pd.DataFrame({k: [v] for k, v in v.items()}, index=[k])
            for k, v in design.items()
        ],
        sort=True,
    )

    if covariates is None:
        covariates = [
            (k, list(map(str, xs.unique().tolist())))
            for k, xs in design_table.iteritems()
        ]

    if len(covariates) == 0:
        return pd.DataFrame(
            np.zeros((0, len(design_table))), columns=design_table.index
        )

    for covariate, values in covariates:
        if covariate not in design_table:
            design_table[covariate] = 0
        design_table[covariate] = design_table[covariate].astype("category")
        design_table[covariate].cat.set_categories(values, inplace=True)
    design_table = design_table[[x for x, _ in covariates]]

    for covariate in design_table:
        if np.any(pd.isna(design_table[covariate])):
            log(
                WARNING, 'Design covariate "%s" has missing values.', covariate
            )

    def _encode(covariate):
        oh_matrix = np.eye(len(covariate.cat.categories), dtype=int)[
            :, covariate.cat.codes
        ]
        oh_matrix[:, covariate.cat.codes == -1] = 0
        return pd.DataFrame(
            oh_matrix, index=covariate.cat.categories, columns=covariate.index
        )

    ks, vs = zip(*[(k, _encode(v)) for k, v in design_table.iteritems()])
    return pd.concat(vs, keys=ks)


def find_device(x: Any) -> torch.device:
    r"""
    Tries to find the :class:`torch.device` associated with the given object
    """

    class NoDevice(Exception):
        # pylint: disable=missing-class-docstring
        pass

    if isinstance(x, torch.Tensor):
        return x.device

    if isinstance(x, list):
        for y in x:
            try:
                return find_device(y)
            except NoDevice:
                pass

    if isinstance(x, dict):
        for y in x.values():
            try:
                return find_device(y)
            except NoDevice:
                pass

    raise NoDevice(f"Failed to find a device associated with {x}")


def remove_fg_elements(mask, size_threshold: float):
    r"""Removes small foreground elements"""
    labels, _ = label(mask)
    labels_unique, label_counts = np.unique(labels, return_counts=True)
    small_labels = labels_unique[
        label_counts < size_threshold ** 2 * np.prod(mask.shape)
    ]
    mask[np.isin(labels, small_labels)] = False
    return mask


def sparseonehot(labels: torch.Tensor, num_classes: Optional[int] = None):
    r"""One-hot encodes a label vectors into a sparse tensor"""
    if num_classes is None:
        num_classes = cast(int, labels.max().item()) + 1
    idx = torch.stack([torch.arange(labels.shape[0]).to(labels), labels])
    return torch.sparse.LongTensor(  # type: ignore
        idx,
        torch.ones(idx.shape[1]).to(idx),
        torch.Size([labels.shape[0], num_classes]),
    )


def isoftplus(x, /):
    r"""
    Inverse softplus.

    >>> ((isoftplus(torch.nn.functional.softplus(torch.linspace(-5, 5)))
    ...     - torch.linspace(-5, 5)) < 1e-5).all()
    tensor(True)
    """
    return np.log(np.exp(x) - 1)


@overload
def to_device(
    x: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: List[Any], device: Optional[torch.device] = None
) -> List[Any]:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: Dict[Any, Any], device: Optional[torch.device] = None
) -> Dict[Any, Any]:
    # pylint: disable=missing-function-docstring
    ...


def to_device(x, device=None):
    r"""
    Converts :class:`torch.Tensor` or a collection of :class:`torch.Tensor` to
    the given :class:`torch.device`
    """
    if device is None:
        device = get("default_device")
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(y, device) for y in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def with_(ctx: ContextManager) -> Callable[[Callable], Callable]:
    r"""
    Creates a decorator that runs the decorated function in the given context
    manager
    """

    def _decorator(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            with ctx:
                return f(*args, **kwargs)

        return _wrapped

    return _decorator
