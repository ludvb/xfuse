from typing import Callable, List, Optional, Union, cast

import numpy as np
import pyro
import torch
from sklearn.decomposition import PCA
from umap import UMAP

from ..data import Data, Dataset
from ..data.slide import FullSlide, Slide
from ..data.utility.misc import make_dataloader
from ..logging import WARNING, log
from ..session import Session, require
from ..utility import center_crop


__all__ = ["reduce_last_dimension", "visualize_factors"]


def visualize_factors(
    method: str = "pca", num_training_samples: Optional[int] = None,
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
        zero_label = torch.where(x["ST"]["data"][0].sum(1) == 0)[0] + 1
        mask = ~np.isin(
            center_crop(x["ST"]["label"][0], activation.shape[:2]), zero_label,
        )
        return activation, mask

    compute_fn = {"ST": _compute_st_activation}

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(data=v.data, iterator=FullSlide)
                    for k, v in dataloader.dataset.data.slides.items()
                },
                design=dataloader.dataset.data.design,
            ),
        ),
        batch_size=1,
    )

    with Session(default_device=torch.device("cpu"), eval=True):
        activations: List[np.ndarray] = []
        masks: List[torch.Tensor] = []
        for x in dataloader:
            data_type, *__this_should_be_empty = x.keys()
            assert __this_should_be_empty == []
            if data_type not in compute_fn:
                log(
                    WARNING,
                    'Factor visualization for data type "%s" not implemented',
                    data_type,
                )
            with torch.no_grad():
                activation, mask = compute_fn[data_type](x)
            activations.append(activation)
            masks.append(mask)

    activations_flat = np.concatenate(
        [activation[mask] for activation, mask in zip(activations, masks)]
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

    for activation, mask in zip(activations, masks):
        yield reduce_last_dimension(
            x=activation, mask=mask, transformation=reduction.transform,
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
