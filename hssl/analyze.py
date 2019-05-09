from functools import reduce

import itertools as it

import operator as op

import os

import re

from typing import List, NamedTuple, Optional

import warnings

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

from tqdm import tqdm

from .image import to_array
from .logging import DEBUG, INFO, WARNING, log
from .network import Histonet, STD
from .utility import integrate_loadings


class Sample(NamedTuple):
    name: str
    image: Image
    label: Image
    data: pd.DataFrame
    effects: np.ndarray


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
            t.abs(std.rgt.distribution.loc) / std.rgt.distribution.scale
        )
        .sum(0)
        .argsort(descending=True)
    )


def analyze(
        histonet: Histonet,
        std: STD,
        sample: Sample,
        output_prefix: str = None,
        device: t.device = None,
):
    if output_prefix is None:
        output_prefix = '.'

    if device is None:
        device = t.device('cpu')

    if sample.label is not None:
        label = (
            t.as_tensor(to_array(sample.label))
            .to(device)
            .permute(2, 0, 1)
            .long()
        )
        label_mask = label[0] != 1
    else:
        label = None
        label_mask = t.ones((sample.image.height, sample.image.width)).byte()

    if sample.data is not None:
        data = (
            t.as_tensor(sample.data.values)
            .to(device)
            .float()
        )
    else:
        data = None

    z, img_distr, loadings, state = histonet(
        (
            t.as_tensor(to_array(sample.image))
            .to(device)
            .permute(2, 0, 1)
            .float()
            [None, ...]
            / 255 * 2 - 1
        ),
        label, data,
    )

    std.resample()
    factors = (
        loadings.exp()
        * (
            (
                (
                    std.rgt.distribution.mean
                    + std.rg.distribution.mean[..., None]
                    + std.lg.distribution.mean[..., None]
                )
                .exp()
            )
            .sum(0)
            [None, ..., None, None]
        )
    )
    mix = factors / factors.sum(1).unsqueeze(1)

    if sample.label is not None:
        label_mask = label[0] != 1
    else:
        label_mask = t.ones((sample.image.height, sample.image.width)).byte()

    for img, prefix in [
            (
                (
                    (
                        img_distr
                        .sample()
                        .clamp(-1, 1)
                        + 1
                    ) / 2
                )[0].permute(1, 2, 0).detach().cpu().numpy(),
                'he-sample',
            ),
            (
                (
                    (img_distr.mean.clamp(-1, 1) + 1) / 2
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
                visualize_greyscale(clip(p, 0.1), label_mask.cpu().numpy()),
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
        sample: Sample,
        patterns: List[str],
        output_prefix: str = None,
        device: t.device = None,
):
    if output_prefix is None:
        output_prefix = '.'

    if device is None:
        device = t.device('cpu')

    if sample.label is not None:
        label = (
            t.as_tensor(to_array(sample.label))
            .to(device)
            .permute(2, 0, 1)
            .long()
        )
        label_mask = label[0] != 1
    else:
        label = None
        label_mask = t.ones((sample.image.height, sample.image.width)).byte()

    if sample.data is not None:
        data = (
            t.as_tensor(sample.data.values)
            .to(device)
            .float()
        )
    else:
        data = None

    z, img_distr, loadings, state = histonet(
        (
            t.as_tensor(to_array(sample.image))
            .to(device)
            .permute(2, 0, 1)
            .float()
            [None, ...]
            / 255 * 2 - 1
        ),
        label, data,
    )

    std.resample()
    gt = (std.rgt.value + std.rg.value[..., None]).exp()
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
            visualize_greyscale(
                clip(gene_map, 0.01),
                label_mask.cpu().numpy(),
            ),
        )
        imwrite(
            os.path.join(relp, f'{std.genes[idx]}.png'),
            visualize_greyscale(
                clip(gene_map / total, 0.01),
                label_mask.cpu().numpy(),
            ),
        )


def impute_counts(
        histonet: Histonet,
        std: STD,
        sample: Sample,
        region: Image,
        device: Optional[t.device] = None,
):
    if device is None:
        device = t.device('cpu')

    if sample.data is not None:
        data = (
            t.as_tensor(sample.data.values)
            .to(device)
            .float()
        )
    else:
        data = None

    label, region = (
        t.as_tensor(to_array(x))
        .to(device)
        .permute(2, 0, 1)
        .long()
        for x in (sample.label, region)
    )

    z, img_distr, loadings, state = histonet(
        (
            t.as_tensor(to_array(sample.image))
            .to(device)
            .permute(2, 0, 1)
            .float()
            [None, ...]
            / 255 * 2 - 1
        ),
        label,
        data,
    )

    region = region.cpu().numpy()[0]
    regions = [*sorted(np.unique(region))]
    region = np.searchsorted(regions, region)

    integrated_loadings = integrate_loadings(
        loadings,
        (
            t.as_tensor(region)
            .to(device)
            .long()
            .unsqueeze(0)
        ),
    )
    d = std.resample()(
        integrated_loadings[2:],
        t.as_tensor(sample.effects).to(device),
    )

    return d, regions[2:]


def dge(
        histonet: Histonet,
        std: STD,
        samples: List[Sample],
        regions: List[Image],
        output: str,
        normalize: bool = True,
        trials: int = 100,
        device: Optional[t.device] = None,
) -> None:
    if device is None:
        device = t.device('cpu')

    def _replace_regions(region, label):
        region[
            (label == 1)
            | ((region != 0) & (region != 255))
        ] = 1
        region[region == 255] = 2

    def _filter_extra_channels(image):
        if len(image.shape) == 3:
            if image.shape[-1] != 1:
                warnings.warn(
                    'label and region images should be single channel. '
                    'only the first channel will be used.'
                )
            image = image[..., 0]
        elif len(image.shape) != 2:
            raise ValueError('invalid image dimensions')
        return image[..., None]

    def _sample_one(sample, region):
        if sample.data is not None:
            data = (
                t.as_tensor(sample.data.values)
                .to(device)
                .float()
            )
        else:
            data = None

        region, image, label = map(
            to_array, (region, sample.image, sample.label))

        region, label = map(_filter_extra_channels, (region, label))
        _replace_regions(region, label)

        region, label = [
            t.as_tensor(x)
            .to(device)
            .permute(2, 0, 1)
            .long()
            for x in (region, label)
        ]
        image = (
            t.as_tensor(image)
            .to(device)
            .permute(2, 0, 1)
            .float()
            [None, ...]
            / 255 * 2 - 1
        )

        _z, _img_distr, loadings, _state = histonet(image, label, data)

        region = region.cpu().numpy()[0]

        integrated_loadings = (
            integrate_loadings(
                loadings,
                (
                    t.as_tensor(region)
                    .to(device)
                    .long()
                    .unsqueeze(0)
                ),
                2,
            )
            [[0, 2]]
        )

        return (
            std(
                integrated_loadings,
                (
                    t.tensor(sample.effects)
                    .to(integrated_loadings)
                    .expand(2, -1)
                ),
            )
            .mean
        )

    def _sample_all():
        std.resample()
        a, b = (
            x / x.sum() for x in
            t.stack([*it.starmap(_sample_one, zip(samples, regions))])
            .sum(0)
        )
        return (t.log(a) - t.log(b)).cpu().detach()

    lfcs = t.stack([_sample_all() for _ in tqdm(
        range(trials),
        dynamic_ncols=True,
    )])

    low, high = t.stack(
        [
            xs.sort()[0][[
                int(len(xs) * .01),
                len(xs) - 1 - int(len(xs) * .01)
            ]]
            for xs in lfcs.transpose(0, 1)
        ],
        -1,
    )

    df = pd.DataFrame(dict(
        gene=std.genes,
        lfc_mean=lfcs.mean(0).numpy(),
        lfc_low=low.numpy(),
        lfc_high=high.numpy(),
        pnorm=(
            t.distributions.Normal(
                lfcs.mean(0).abs(),
                lfcs.std(0),
            )
            .cdf(0)
        ),
    ))
    (
        df
        .sort_values('lfc_mean')
        .to_csv(
            os.path.join(output, 'dge.csv.gz'),
            index=False,
        )
    )

    df['neg_log_cv'] = (
        (t.log(lfcs.mean(0).abs()) - t.log(lfcs.std(0)))
        .numpy()
    )
    plot = (
        pn.ggplot(df)
        + pn.aes(
            'lfc_mean',
            'neg_log_cv',
            xmin='lfc_low',
            xmax='lfc_high',
        )
        + pn.geom_point(alpha=0.3)
        + pn.xlab('log-fold change')
        + pn.ylab('-log CV')
        + pn.geom_text(
            pn.aes('lfc_mean', 'neg_log_cv', label='gene'),
            (
                pd.concat([
                    (
                        df[df.lfc_mean < -1]
                        .sort_values('lfc_mean')
                        .iloc[:10]
                    ),
                    (
                        df[df.lfc_mean > 1]
                        .sort_values('lfc_mean', ascending=False)
                        .iloc[:10]
                    ),
                ])
            ),
            color='black',
            nudge_y=0.3,
        )
        + pn.geom_errorbarh(height=None, alpha=0.05)
        + pn.theme_bw()
        + pn.theme(legend_position='none')
    )
    plot.save(os.path.join(output, 'volcano.pdf'), width=20, height=20)

    data = df.sort_values('lfc_mean')
    data = pd.concat([data.iloc[:50], data.iloc[-50:]])
    data.gene = data.gene.astype(CategoricalDtype(data.gene, ordered=True))
    plot = (
        pn.ggplot(data)
        + pn.aes('gene', 'lfc_mean', ymax='lfc_low', ymin='lfc_high')
        + pn.geom_point()
        + pn.geom_errorbar()
        + pn.ylab('log-fold change')
        + pn.coord_flip()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=90),
        )
    )
    plot.save(os.path.join(output, 'top-dg.pdf'), width=10, height=20)

    for sample, region in zip(samples, regions):
        region, image, label = map(
            to_array, (region, sample.image, sample.label))
        region, label = map(_filter_extra_channels, (region, label))
        _replace_regions(region, label)
        annotation = (
            image / 255 / 2
            + 1 / 2 * np.concatenate([
                (region == 0),
                np.zeros((*region.shape[:2], 2)),
            ], 2)
            + 1 / 2 * np.concatenate([
                np.zeros((*region.shape[:2], 2)),
                (region == 2),
            ], 2)
        )
        imwrite(
            os.path.join(output, f'{sample.name}_annotation.png'),
            annotation,
        )
