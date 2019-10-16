from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyvips import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid


class Sample(NamedTuple):
    # pylint: disable=missing-class-docstring
    name: str
    image: Image
    label: Image
    data: pd.DataFrame
    effects: np.ndarray


def _run_tsne(y, n_components=3, initial_dims=20):
    x = y.reshape(-1, y.shape[-1])

    def uniformize(x):
        return (x - x.min(0)) / (x.max(0) - x.min(0))

    print("performing PCA")
    pca_map = PCA(n_components=initial_dims).fit_transform(x)
    print("performing tSNE")
    tsne_map = TSNE(n_components=n_components, verbose=1).fit_transform(
        pca_map
    )
    tsne_map = uniformize(tsne_map)
    return tsne_map.reshape((*y.shape[:2], -1))


def _visualize(model, z):
    nmu, *_ = model.decode(z)
    return plt.imshow(nmu[0].detach().numpy().transpose(1, 2, 0))


def _interpolate(model, a, b, output_file="/tmp/interpolation.mp4"):
    from matplotlib.animation import ArtistAnimation

    fig = plt.figure()
    anim = ArtistAnimation(
        fig,
        [[_visualize(model, a + (b - a) * k)] for k in np.linspace(0, 1, 100)],
        repeat_delay=1000,
        interval=50,
        blit=True,
    )
    anim.save(output_file)


def _side_by_side(x, y):
    plt.subplot(1, 2, 1)
    plt.imshow(x["image"][0].permute(1, 2, 0).detach())
    plt.subplot(1, 2, 2)
    pca = (
        PCA(n_components=3)
        .fit_transform(y[0].reshape(y.shape[1], -1).t().detach())
        .reshape(*y.shape[-2:], -1)
    )
    _min, _max = np.quantile(pca, [0.1, 0.9])
    pca = pca.clip(_min, _max)
    plt.imshow((pca - _min) / (_max - _min))
    plt.show()


def _clip(x, q):
    if isinstance(q, float):
        minq, maxq = q, 1 - q
    else:
        try:
            minq, maxq = q
        except TypeError:
            raise ValueError("`q` mus be float or iterable")
    return np.clip(x, *np.quantile(x, [minq, maxq]))


def _normalize(x):
    _min = x.min((0, 2, 3))[..., None, None]
    _max = x.max((0, 2, 3))[..., None, None]
    return (x - _min) / (_max - _min)


def _visualize_greyscale(image, mask=None):
    if mask is None:
        mask = np.ones_like(image)
    return (
        (
            1
            - ((image - image.min()) / (image.max() - image.min()) + 0.1)
            / 1.1
            * mask
        )
        * 255
    ).astype(np.uint8)


def _visualize_batch(batch, normalize=False, **kwargs):
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu()
    else:
        batch = torch.as_tensor(batch)
    return np.transpose(
        (
            make_grid(
                batch,
                nrow=int(np.floor(np.sqrt(len(batch)))),
                padding=int(
                    np.ceil(np.sqrt(np.product(batch.shape[-2:])) / 100)
                ),
                normalize=normalize,
                **kwargs,
            )
            .detach()
            .cpu()
            .numpy()
        ),
        (1, 2, 0),
    )


def impute(sample, region):
    """Imputation analysis"""
    raise NotImplementedError()


def dge(samples, regions, output, normalize, trials):
    """Diffential gene expression analysis"""
    raise NotImplementedError()
