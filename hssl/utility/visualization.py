from typing import Optional, Union, cast

import numpy as np
import torch
from sklearn.decomposition import PCA

__all__ = ["reduce_last_dimension"]


def reduce_last_dimension(
    x: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = "pca",
    n_components: int = 3,
    **kwargs,
) -> np.ndarray:
    r"""
    Performs dimensionality reduction on the last dimension of the input array
    """

    if method != "pca":
        raise NotImplementedError()

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=np.bool)
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)
    mask = cast(np.ndarray, mask)

    if x.shape[-1] == 1 and n_components == 1:
        return mask.astype(np.float32)[..., None]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    values = PCA(n_components=n_components, **kwargs).fit_transform(x[mask])

    dst = np.zeros((*mask.shape, n_components))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst
