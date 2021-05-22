import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pyro
import torch
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.decomposition import PCA

from ..data import Data, Dataset
from ..data.slide import FullSlideIterator, Slide
from ..data.utility.misc import make_dataloader
from ..session import Session, get, require
from ..utility.core import center_crop
from ..utility.mask import cleanup_mask


__all__ = ["reduce_last_dimension", "visualize_metagenes"]


def _cmyk2rgb(x: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(x, mode="CMYK").convert("RGB"))


def _normalize(x: np.ndarray, axis=None) -> np.ndarray:
    x = x - x.min(axis, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = x / x.max(axis, keepdims=True)
        x = np.nan_to_num(x)
    return x


def balance_colors(
    x: np.ndarray, q: float = 0.01, q_high: Optional[float] = None, axis=None
) -> np.ndarray:
    r"""
    Balances colors by shifting quantiles `q` and `q_high` to 0.0 and 1.0 using
    an affine transformation and clipping all values to [0.0, 1.0].
    If `x` is np.uint8, the range [0, 255] is used instead.

    >>> balance_colors(np.array([1.0, 2.0, 4.0, 5.0]))
    array([0.  , 0.25, 0.75, 1.  ])
    >>> balance_colors(np.array([1, 2, 4, 5], dtype=np.uint8))
    array([  0,  64, 191, 255], dtype=uint8)
    """
    if q_high is None:
        q_high = 1.0 - q
    xmin, xmax = np.quantile(
        x, [q, q_high], axis=axis, interpolation="nearest"
    )
    y = x.clip(xmin, xmax)
    y = _normalize(y, axis)
    if x.dtype == np.uint8:
        y = (255 * y).round().astype(np.uint8)
    return y


def greyscale2colormap(x: np.ndarray) -> np.ndarray:
    r"""
    Applies the current :class:`Session` `colormap` to greyscale image `x`.
    """
    colormap = get("colormap")
    if x.ndim != 2:
        raise ValueError(
            f"Image must have exactly two dimensions (got {x.ndim=})."
        )
    if x.dtype != np.uint8:
        x = _normalize(x)
        x = (255 * x).round().astype(np.uint8)
    return np.round(255 * np.array(colormap.colors)[x]).astype(np.uint8)


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
) -> Iterable[Tuple[str, np.ndarray, List[Tuple[str, np.ndarray]]]]:
    r"""Creates visualizations of metagenes"""

    # pylint: disable=too-many-locals,too-many-statements

    model = require("model")
    dataloader = require("dataloader")

    if method not in ["pca"]:
        raise ValueError(f'Method "{method}" not supported')

    def _compute_st_activation(x):
        x["ST"]["label"].zero_()

        with Session(default_device=torch.device("cpu")):
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
            model_trace.trace.nodes["scale"]["fn"].mean[0].squeeze().numpy()
        )
        scale = scale / scale.max()

        zero_label = np.isin(
            center_crop(x["ST"]["label"][0], activation.shape[:2]),
            torch.where(x["ST"]["data"][0].sum(1) == 0)[0] + 1,
        )
        max_val = (
            model_trace.trace.nodes["scale"]["value"]
            .flatten()
            .kthvalue(
                int(
                    round(
                        0.95
                        * model_trace.trace.nodes["scale"]["value"].numel()
                    )
                )
            )
            .values
        )
        thresholded_scale = (
            model_trace.trace.nodes["scale"]["value"] > 0.01 * max_val
        )
        thresholded_scale = thresholded_scale.squeeze().cpu().numpy()
        mask = ~zero_label & thresholded_scale
        mask = cleanup_mask(mask, 0.01)

        metagene_name = list(model.get_experiment("ST").metagenes.keys())

        return activation, scale, mask, metagene_name

    compute_fn = {"ST": _compute_st_activation}

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(data=v.data, iterator=FullSlideIterator)
                    for k, v in dataloader.dataset.data.slides.items()
                },
                design=dataloader.dataset.data.design,
            )
        ),
        batch_size=1,
    )

    with Session(eval=True):
        activations, scales, masks, metagene_names, slide_names = zip(
            *[
                (*compute_fn[datatype](batch), batch[datatype]["slide"][0])
                for batch in dataloader
                for datatype in batch
            ]
        )

    activations_flat = np.concatenate(
        [
            activation[scale > 0.01]
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
        reduction = PCA(n_components=min(3, activations_training.shape[-1]))
    else:
        raise RuntimeError("This path should not be reachable")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            a = reduction.fit_transform(activations_training)

            def transform(x):
                x = reduction.transform(x)
                x = (x - a.min(0)) / (a.max(0) - a.min(0))
                x = x.clip(0.0, 1.0)
                return x

        except RuntimeWarning:
            transform = lambda x: x[..., :3]

    for activation, scale, mask, metagene_name, slide_name in zip(
        activations, scales, masks, metagene_names, slide_names
    ):
        summarized_image = np.zeros((*activation.shape[:2], 4), dtype=float)
        summarized_activation = reduce_last_dimension(
            x=activation, transformation=transform
        )
        summarized_image[
            ...,
            # pylint: disable=unsubscriptable-object
            : summarized_activation.shape[-1],
        ] = _normalize(summarized_activation)
        summarized_image[..., -1] = balance_colors(
            1.0 - scale, q=0.05, q_high=1.0
        )
        summarized_image = np.round(255 * summarized_image)
        summarized_image = summarized_image.astype(np.uint8)
        summarized_image = _cmyk2rgb(summarized_image)
        yield (
            slide_name,
            mask_background(summarized_image, mask),
            [
                (
                    n,
                    mask_background(
                        greyscale2colormap(
                            balance_colors(scale * x, q=0.0, q_high=0.995)
                        ),
                        mask,
                    ),
                )
                for n, x in zip(metagene_name, activation.transpose(2, 0, 1))
            ],
        )


def reduce_last_dimension(
    x: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    transformation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    r"""
    Performs dimensionality reduction on the last dimension of the input array
    """

    def _default_transformation(x):
        return PCA(n_components=min(x.shape[-1], 3)).fit_transform(x)

    if transformation is None:
        transformation = _default_transformation

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=bool)
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)
    mask = cast(np.ndarray, mask)

    if isinstance(x, torch.Tensor):
        x = cast(np.ndarray, x.detach().cpu().numpy())

    values = transformation(x[mask])
    dst = np.zeros((*mask.shape, values.shape[-1]))
    dst[mask] = _normalize(values, axis=0)

    return dst
