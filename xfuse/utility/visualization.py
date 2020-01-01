from typing import Callable, List, Optional, Union, cast

import numpy as np
import pyro
import torch
from PIL import Image
from sklearn.decomposition import PCA
from umap import UMAP

from ..data import Data, Dataset
from ..data.slide import FullSlide, Slide
from ..data.utility.misc import make_dataloader
from ..logging import WARNING, log
from ..session import Session, require


__all__ = ["reduce_last_dimension", "visualize_metagenes"]


def _cmyk2rgb(x: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(x, mode="CMYK").convert("RGB"))


def visualize_metagenes(
    method: str = "pca", num_training_samples: Optional[int] = None
) -> np.ndarray:
    r"""Creates visualizations of metagenes"""
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
        return activation, scale

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
                activation, scale = compute_fn[data_type](x)
            activations.append(activation)
            scales.append(scale)

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

    for activation, scale in zip(activations, scales):
        summarized_activations = reduce_last_dimension(
            x=activation, transformation=reduction.transform
        )
        summarized_activations = np.concatenate(
            [summarized_activations, 1.0 - scale], axis=-1
        )
        summarized_activations = np.round(255 * summarized_activations)
        summarized_activations = summarized_activations.astype(np.uint8)
        yield _cmyk2rgb(summarized_activations)


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
    dst = np.zeros((*mask.shape, 3))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst
