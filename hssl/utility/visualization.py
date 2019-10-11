from typing import Optional, Union

import numpy as np

import torch as t


__all__ = ["reduce_last_dimension"]


def reduce_last_dimension(
    x: Union[t.Tensor, np.ndarray],
    mask: Optional[Union[t.Tensor, np.ndarray]] = None,
    method: str = "pca",
    n_components: int = 3,
    **kwargs,
) -> np.ndarray:
    if method != "pca":
        raise NotImplementedError()

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=bool)
    elif isinstance(mask, t.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)

    from sklearn.decomposition import PCA

    if isinstance(x, t.Tensor):
        x = x.detach().cpu().numpy()

    values = PCA(n_components=n_components, **kwargs).fit_transform(x[mask])

    dst = np.zeros((*mask.shape, n_components))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst
