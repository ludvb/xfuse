from typing import Callable, List, Optional, Union, cast

import numpy as np
import pyro
import torch
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from sklearn.decomposition import PCA
from umap import UMAP

from ..data import Data, Dataset
from ..data.slide import FullSlide, Slide
from ..data.utility.misc import make_dataloader
from ..logging import WARNING, log
from ..session import Session, require
from ..utility import center_crop


__all__ = ["reduce_last_dimension", "visualize_metagenes"]


def _cmyk2rgb(x: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(x, mode="CMYK").convert("RGB"))


def balance_colors(
    x: np.ndarray, q: float = 0.01, q_high: Optional[float] = None
) -> np.ndarray:
    r"""
    Balances colors by shifting quantiles `q` and `q_high` to 0.0 and 1.0 using
    a per channel affine transformation and clipping all values to [0.0, 1.0].
    If `x` is np.uint8, the range [0, 255] is used instead.

    >>> balance_colors(np.array([1.0, 2.0, 4.0, 5.0]))
    array([0.  , 0.25, 0.75, 1.  ])
    >>> balance_colors(np.array([1, 2, 4, 5], dtype=np.uint8))
    array([  0,  64, 191, 255], dtype=uint8)
    """
    if q_high is None:
        q_high = 1.0 - q
    if x.ndim > 2:
        y = x.reshape(-1, x.shape[-1])
    else:
        y = x.flatten()
    ymin, ymax = np.quantile(y, [q, q_high], axis=0, interpolation="nearest")
    y = (y - ymin) / (ymax - ymin)
    y = y.clip(0.0, 1.0)
    y = y.reshape(x.shape)
    if x.dtype == np.uint8:
        y = (255 * y).round().astype(np.uint8)
    return y


def mask_background(
    image: np.ndarray,
    mask: np.ndarray,
    border: int = 5,
    border_color: Optional[np.ndarray] = None,
    background_color: Optional[np.ndarray] = None,
):
    r"""
    Masks out background elements and adds a border to the non-masked elements.

    >>> mask_background(image=100 * np.eye(3, dtype=np.uint8),
    ...            mask=np.eye(3, dtype=bool), border=1)
    array([[100,   0, 255],
           [  0, 100,   0],
           [255,   0, 100]], dtype=uint8)
    """
    if border_color is None:
        if image.ndim == 2:
            border_color = np.array(0, dtype=np.uint8)
        else:
            border_color = np.zeros(image.shape[-1], dtype=np.uint8)
    if background_color is None:
        if image.ndim == 2:
            background_color = np.array(255, dtype=np.uint8)
        else:
            background_color = 255 * np.ones(image.shape[-1], dtype=np.uint8)

    image = image.copy()
    image[~mask] = background_color
    border_mask = distance_transform_edt(~mask) <= border
    border_mask &= ~mask
    image[border_mask] = border_color
    return image


def visualize_metagenes(
    method: str = "pca", num_training_samples: Optional[int] = None
) -> np.ndarray:
    r"""Creates visualizations of metagenes"""

    # pylint: disable=too-many-statements

    model = require("model")
    dataloader = require("dataloader")

    if method not in ["pca", "umap"]:
        raise ValueError(f'Method "{method}" not supported')

    def _compute_st_activation(x):
        with pyro.poutine.trace() as guide_trace:
            model.guide(x)
        with pyro.poutine.replay(trace=guide_trace.trace):
            with pyro.poutine.trace() as model_trace:
                model.model(x)
        activation = (
            model_trace.trace.nodes["rim"]["fn"]
            .mean[0]
            .permute(1, 2, 0)
            .numpy()
        )
        scale = (
            model_trace.trace.nodes["scale"]["fn"]
            .mean[0]
            .permute(1, 2, 0)
            .numpy()
        )
        scale = scale / scale.max()
        zero_label = torch.where(x["ST"]["data"][0].sum(1) == 0)[0] + 1
        mask = ~np.isin(
            center_crop(x["ST"]["label"][0], activation.shape[:2]), zero_label
        )
        mask &= scale.squeeze() > 0.01
        return activation, scale, mask

    compute_fn = {"ST": _compute_st_activation}

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(data=v.data, iterator=FullSlide)
                    for k, v in dataloader.dataset.data.slides.items()
                },
                design=dataloader.dataset.data.design,
            )
        ),
        batch_size=1,
    )

    with Session(default_device=torch.device("cpu"), eval=True):
        activations: List[np.ndarray] = []
        scales: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for x in dataloader:
            data_type, *__this_should_be_empty = x.keys()
            assert __this_should_be_empty == []
            if data_type not in compute_fn:
                log(
                    WARNING,
                    'Metagene visualization for data type "%s" not'
                    " implemented",
                    data_type,
                )
            with torch.no_grad():
                activation, scale, mask = compute_fn[data_type](x)
            activations.append(activation)
            scales.append(scale)
            masks.append(mask)

    activations_flat = np.concatenate(
        [
            activation[scale.squeeze() > 0.01]
            for activation, scale in zip(activations, scales)
        ]
    )
    if num_training_samples:
        activations_training = activations_flat[
            np.random.choice(
                # pylint: disable=unsubscriptable-object
                activations_flat.shape[0],
                min(activations_flat.shape[0], num_training_samples),
                replace=False,
            )
        ]
    else:
        activations_training = activations_flat

    if method == "pca":
        reduction = PCA(n_components=3)
    elif method == "umap":
        reduction = UMAP(n_components=3)
    else:
        raise RuntimeError("This path should not be reachable")

    reduction.fit(activations_training)

    for activation, scale, mask in zip(activations, scales, masks):
        summarized_activations = reduce_last_dimension(
            x=activation, transformation=reduction.transform
        )
        summarized_activations = np.concatenate(
            [summarized_activations, 1.0 - scale], axis=-1
        )
        summarized_activations = np.round(255 * summarized_activations)
        summarized_activations = summarized_activations.astype(np.uint8)
        summarized_activations = _cmyk2rgb(summarized_activations)
        summarized_activations = balance_colors(summarized_activations)
        summarized_activations = mask_background(
            summarized_activations, binary_fill_holes(mask)
        )
        yield summarized_activations


def reduce_last_dimension(
    x: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    transformation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    r"""
    Performs dimensionality reduction on the last dimension of the input array
    """

    def _default_transformation(x):
        return PCA(n_components=3).fit_transform(x)

    if transformation is None:
        transformation = _default_transformation

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=np.bool)
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)
    mask = cast(np.ndarray, mask)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    values = transformation(x[mask])
    dst = np.zeros((*mask.shape, values.shape[-1]))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst
