from functools import reduce

import itertools as it

import os

import re

from typing import List, Optional

from dfply import X, mutate

from imageio import imwrite

import plotnine as pn

from pyvips import Image

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype

import torch as t

from torchvision.utils import make_grid

from .image import to_array
from .logging import DEBUG, INFO, WARNING, log
from .network import Histonet, STD


def run_tsne(y, n_components=3, initial_dims=20):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = y.reshape(-1, y.shape[-1])

    def uniformize(x):
        return (x - x.min(0)) / (x.max(0) - x.min(0))

    print("performing PCA")
    pca_map = PCA(n_components=initial_dims).fit_transform(x)
    print("performing tSNE")
    tsne_map = (
        TSNE(n_components=n_components, verbose=1)
        .fit_transform(pca_map)
    )
    tsne_map = uniformize(tsne_map)
    return tsne_map.reshape((*y.shape[:2], -1))


def visualize(model, z):
    import matplotlib.pyplot as plt
    nmu, nsd, *_ = model.decode(z)
    return plt.imshow(nmu[0].detach().numpy().transpose(1, 2, 0))


def interpolate(model, z1, z2, to='/tmp/interpolation.mp4'):
    from matplotlib.animation import ArtistAnimation
    fig = plt.figure()
    anim = ArtistAnimation(
        fig,
        [
            [visualize(model, z1 + (z2 - z1) * k)]
            for k in np.linspace(0, 1, 100)
        ],
        repeat_delay=1000,
        interval=50,
        blit=True,
    )
    anim.save(to)


def side_by_side(x, y):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    plt.subplot(1, 2, 1)
    plt.imshow(x['image'][0].permute(1, 2, 0).detach())
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


def clip(x, q):
    if isinstance(q, float):
        minq, maxq = q, 1 - q
    else:
        try:
            minq, maxq = q
        except TypeError:
            raise ValueError('`q` mus be float or iterable')
    return np.clip(x, *np.quantile(x, [minq, maxq]))


def normalize(x):
    _min = x.min((0, 2, 3))[..., None, None]
    _max = x.max((0, 2, 3))[..., None, None]
    return (x - _min) / (_max - _min)


def visualize_greyscale(im, mask=None):
    if mask is None:
        mask = np.ones_like(im)
    return (
        (1 - ((im - im.min()) / (im.max() - im.min()) + 0.1) / 1.1
            * mask)
        * 255
    ).astype(np.uint8)


def dim_red(x, mask=None, method='pca', n_components=3, **kwargs):
    if method != 'pca':
        raise NotImplementedError()

    if mask is None:
        mask = np.ones(x.shape[:-1], dtype=bool)
    elif isinstance(mask, t.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)

    from sklearn.decomposition import PCA

    if isinstance(x, t.Tensor):
        x = x.detach().cpu().numpy()

    values = (
        PCA(n_components=n_components, **kwargs)
        .fit_transform(x[mask])
    )

    dst = np.zeros((*mask.shape, n_components))
    dst[mask] = (values - values.min(0)) / (values.max(0) - values.min(0))

    return dst


def visualize_batch(batch, normalize=False, **kwargs):
    if isinstance(batch, t.Tensor):
        batch = batch.detach().cpu()
    else:
        batch = t.as_tensor(batch)
    return np.transpose(
        (
            make_grid(
                batch,
                nrow=int(np.floor(np.sqrt(len(batch)))),
                padding=int(np.ceil(np.sqrt(
                    np.product(batch.shape[-2:])) / 100)),
                normalize=normalize,
            )
            .detach()
            .cpu()
            .numpy()
        ),
        (1, 2, 0),
    )


def order_factors(std: STD):
    return (
        (
            t.abs(std.rgt_q.loc) / std.rgt_q.scale
        )
        .sum(0)
        .argsort(descending=True)
    )


def analyze(
        histonet: Histonet,
        std: STD,
        image: Image,
        label: Optional[Image],
        output_prefix: str = None,
        device: t.device = None,
):
    if output_prefix is None:
        output_prefix = '.'

    if device is None:
        device = t.device('cpu')

    z, mu, sd, loadings, state = histonet(
        t.as_tensor(to_array(image))
        .to(device)
        .permute(2, 0, 1)
        .float()
        [None, ...]
        / 255 * 2 - 1
    )

    std.resample()
    factors = (
        loadings.exp()
        * (
            (std.rate_gt * std.logit.exp()[..., None])
            .sum(0)
            [None, ..., None, None]
        )
    )
    mix = factors / factors.sum(1).unsqueeze(1)

    if label:
        label_mask = to_array(label)[..., 0] != 1
    else:
        label_mask = np.ones((image.height, image.width), dtype=bool)

    for img, prefix in [
            (
                (
                    (
                        t.distributions.Normal(mu, sd)
                        .sample()
                        .clamp(-1, 1)
                        + 1
                    ) / 2
                )[0].permute(1, 2, 0).detach().cpu().numpy(),
                'he-sample',
            ),
            (
                (
                    (mu.clamp(-1, 1) + 1) / 2
                )[0].permute(1, 2, 0).detach().cpu().numpy(),
                'he-mean',
            ),
            (dim_red(z[0].permute(1, 2, 0)), 'z'),
            (dim_red(factors[0].permute(1, 2, 0), label_mask), 'fct-abs'),
            (dim_red(mix[0].permute(1, 2, 0), label_mask), 'fct-rel'),
            (dim_red(state[0].permute(1, 2, 0), label_mask), 'state'),
    ]:
        imwrite(
            os.path.join(output_prefix, f'{prefix}.png'),
            (img * 255).astype(np.uint8),
        )

    for name, data in (
            ('relative', mix),
            ('absolute', factors),
    ):
        os.makedirs(os.path.join(output_prefix, 'factors', name))
        for i, p in enumerate(
                (data[0, f].detach().cpu().numpy()
                 for f in order_factors(std)),
                1,
        ):
            imwrite(
                os.path.join(
                    output_prefix, 'factors', name, f'factor-{i:03d}.png'),
                visualize_greyscale(clip(p, 0.1), label_mask),
            )


def analyze_gene_profiles(
        std: STD,
        genes: List[str],
        factors: Optional[List[int]] = None,
        truncate: Optional[int] = None,
        regex: bool = True,
        output_prefix: str = None,
):
    def _to_df(x, name):
        return (
            pd.DataFrame(x.detach().cpu().numpy(), index=std.genes)
            .rename_axis('gene')
            .reset_index()
            .melt('gene', var_name='factor', value_name=name)
        )

    data = pd.merge(
        _to_df(std.rgt.distribution.loc, 'mu'),
        _to_df(std.rgt.distribution.scale, 'sd'),
        on=['gene', 'factor'],
    )
    if factors is not None:
        data = data.loc[data.factor.isin(factors)]

    data.to_csv(
        os.path.join(output_prefix, 'profiles.csv'),
        index=False,
    )

    def _generate_barplot(x, name):
        return (
            pn.ggplot(
                x
                >> mutate(
                    mu_min=X.mu - 2 * X.sd,
                    mu_max=X.mu + 2 * X.sd,
                )
            )
            + pn.aes('gene', 'mu', ymax='mu_max', ymin='mu_min')
            + pn.geom_point()
            + pn.geom_errorbar()
            + pn.ylab('log fold effect')
            + pn.coord_flip()
            + pn.ggtitle(name)
            + pn.theme(
                axis_text_x=pn.element_text(rotation=90),
                dpi=100,
                figure_size=(7, 7),
            )
        )

    with PdfPages(os.path.join(output_prefix, f'profiles.pdf')) as pdf:
        factor_order = order_factors(std)
        for i, f in enumerate(order_factors(std), 1):
            log(DEBUG, 'producing profile for factor %d', i)

            x = (
                data[data.factor == factor_order[f]]
                .sort_values('mu', ascending=False)
            )
            if genes != []:
                x = x.loc[reduce(
                    lambda a, x: a | x,
                    (x.gene.str.contains(g, regex=regex, flags=re.IGNORECASE)
                     for g in genes),
                )]
            if truncate is not None and len(x) > 2 * truncate:
                x = pd.concat([x.iloc[:truncate], x.iloc[-truncate:]])
            x.gene = x.gene.astype(CategoricalDtype(x.gene, ordered=True))

            _generate_barplot(x, f'factor {i}').draw(False)
            pdf.savefig()
            plt.close()


def analyze_genes(
        histonet: Histonet,
        std: STD,
        image: Image,
        label: Image,
        patterns: List[str],
        output_prefix: str = None,
        device: t.device = None,
):
    if output_prefix is None:
        output_prefix = '.'

    if device is None:
        device = t.device('cpu')

    if label:
        label_mask = to_array(label)[..., 0] != 1
    else:
        label_mask = np.ones((image.height, image.width), dtype=bool)

    z, mu, sd, loadings, state = histonet(
        t.as_tensor(to_array(image))
        .to(device)
        .permute(2, 0, 1)
        .float()
        [None, ...]
        / 255 * 2 - 1
    )

    gt = std.resample().rate_gt
    total = (
        (loadings[0].exp() * gt.sum(0)[..., None, None])
        .sum(0).detach().cpu().numpy()
    )

    idxs, ns = zip(*[
        (xs, len(xs)) for xs in [
            [idx for idx, g in enumerate(std.genes)
             if re.match(p.lower(), g.lower())]
            for p in patterns
        ]
    ])
    for p in (p for p, n in zip(patterns, ns) if n == 0):
        log(WARNING, 'pattern %p did not match any gene', p)

    relp = os.path.join(output_prefix, 'gene-maps', 'relative')
    absp = os.path.join(output_prefix, 'gene-maps', 'absolute')
    os.makedirs(relp)
    os.makedirs(absp)
    for idx in set(it.chain(*idxs)):
        log(INFO, 'creating gene map for %s', std.genes[idx])
        gene_map = (
            t.einsum('txy,t->xy', loadings[0].exp(), gt[idx])
            .detach()
            .cpu()
            .numpy()
        )
        imwrite(
            os.path.join(absp, f'{std.genes[idx]}.png'),
            visualize_greyscale(clip(gene_map, 0.01), label_mask),
        )
        imwrite(
            os.path.join(relp, f'{std.genes[idx]}.png'),
            visualize_greyscale(clip(gene_map / total, 0.01), label_mask),
        )
