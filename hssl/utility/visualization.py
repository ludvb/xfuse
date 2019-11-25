from typing import Optional, Union, cast

import numpy as np
import torch
from sklearn.decomposition import PCA
from umap import UMAP

__all__ = ["reduce_last_dimension"]


def reduce_last_dimension(
    x: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = "umap",
    n_components: int = 3,
    num_training_samples: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Performs dimensionality reduction on the last dimension of the input array
    """
    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=np.bool)
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)
    mask = cast(np.ndarray, mask)

    if x.shape[-1] == 1 and n_components == 1:
        return mask.astype(np.float32)[..., None]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = x[mask]

    if num_training_samples:
        training_samples = np.random.choice(
            x.shape[0], min(x.shape[0], num_training_samples), replace=False,
        )
    else:
        training_samples = np.arange(x.shape[0])

    if method == "pca":
        values = (
            PCA(n_components=n_components, **kwargs)
            .fit(x[training_samples])
            .transform(x)
        )
    elif method == "umap":
        values = (
            UMAP(n_components=n_components, **kwargs)
            .fit(x[training_samples])
            .transform(x)
        )
    else:
        raise NotImplementedError(
            f'Dimensionality reduction method "{method}" not implemented'
        )

    dst = np.zeros((*mask.shape, n_components))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst
