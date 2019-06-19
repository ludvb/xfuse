from abc import abstractmethod

from copy import deepcopy

from functools import reduce

import itertools as it

import operator as op

from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)

import numpy as np

import pyro as p
from pyro.contrib.autoname import scope
import pyro.distributions as distr

import torch as t

from .logging import DEBUG, INFO, log
from .utility import center_crop


def _find_device(x):
    # TODO: move elsewhere
    if isinstance(x, t.Tensor):
        return x.device
    if isinstance(x, list):
        for y in x:
            device = _find_device(y)
            if device is not None:
                return device
    if isinstance(x, dict):
        for y in x.values():
            device = _find_device(y)
            if device is not None:
                return device
    return None


class ExperimentType(t.nn.Module):
    def __init__(self, n: int):
        self.n = n

    def _sample_global(name, *args, **kwargs):
        try:
            return p.sample(name, *args, **kwargs)
        except RuntimeError:
            return p.poutine.runtime._PYRO_STACK[-1].trace.nodes[name]['value']

    @property
    @abstractmethod
    def tag(self):
        pass

    @abstractmethod
    def model(self, x, z):
        pass

    @abstractmethod
    def guide(self, x):
        pass


class ImagingExperiment(ExperimentType):
    @property
    def tag(self):
        return 'image'

    def _decode(self, z):
        decoder = p.module(
            'img_decoder',
            t.nn.Sequential(
                t.nn.Conv2d(z.shape[1], 512, 5, padding=5),
                # x16
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(512),
                Unpool(512, 256, 5),
                # x8
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(256),
                Unpool(256, 128, 5),
                # x4
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(128),
                Unpool(128, 64, 5),
                # x2
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(64),
                Unpool(64, 32, 5),
                # x1
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(32),
            ),
            update_module_params=True,
        ).to(z)
        return decoder(z)

    def _sample_image(self, x, decoded):
        img_mu = p.module(
            'img_mu',
            t.nn.Sequential(
                t.nn.Conv2d(32, 16, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16),
                t.nn.Conv2d(16, 3, 3, 1, 1),
                t.nn.Tanh(),
            ),
            update_module_params=True,
        ).to(decoded)
        img_sd = p.module(
            'img_sd',
            t.nn.Sequential(
                t.nn.Conv2d(32, 16, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16),
                t.nn.Conv2d(16, 3, 3, 1, 1),
                t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)
        mu = center_crop(img_mu(decoded), [None, None, *x['image'].shape[-2:]])
        sd = center_crop(img_sd(decoded), [None, None, *x['image'].shape[-2:]])

        return p.sample(
            'image',
            distr.Normal(mu, sd).to_event(3),
            obs=x['image'],
        )

    def model(self, x, z):
        decoded = self._decode(z)
        with p.poutine.scale(self.n/len(x)):
            self._sample_image(x, decoded)

    def guide(self, x):
        encoder = p.module(
            'img_encoder',
            t.nn.Sequential(
                # x1
                t.nn.Conv2d(3, 64, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(64),
                # x2
                t.nn.Conv2d(64, 128, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(128),
                # x4
                t.nn.Conv2d(128, 256, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(256),
                # x8
                t.nn.Conv2d(256, 512, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(512),
                # x16
            ),
            update_module_params=True,
        ).to(x['image'])
        return encoder(x['image'])


class STExperiment(ImagingExperiment):
    @property
    def tag(self):
        return 'ST'

    def __init__(
            self,
            *args,
            num_factors: int = 1,
            default_scale: float = 1.,
            gene_baseline: Optional[t.Tensor] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__factors = set()
        self.__factors_counter = it.count()
        for _ in range(num_factors):
            self.add_factor()

        self.__default_scale = default_scale

        if gene_baseline is not None:
            p.get_param_store().setdefault(
                'rg',
                gene_baseline.clone().detach())

    @property
    def factors(self):
        return deepcopy(self.__factors)

    def add_factor(self):
        n = next(self.__factors_counter)
        log(DEBUG, 'adding new factor: %d', n)
        self.__factors.add(n)
        return self

    def remove_facor(self, n):
        log(DEBUG, 'removing factor: %d', n)
        try:
            self.__factors.remove(n)
        except KeyError:
            raise ValueError(
                f'attempted to remove factor {n}, which doesn\'t exist!')
        self.__factors_counter = it.chain([n], self.__factors_counter)

    def _get_scale_decoder(self, in_channels):
        decoder = t.nn.Sequential(
            t.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(in_channels),
            t.nn.Conv2d(in_channels, 1, 1, 1, 1),
            t.nn.Softplus(),
        )
        t.nn.init.constant_(decoder[-2].weight, 0.)
        t.nn.init.constant_(
            decoder[-2].bias,
            np.log(np.exp(self.__default_scale) - 1),
        )
        return p.module('scale', decoder, update_module_params=True)

    def _get_factor_decoder(self, in_channels, n):
        decoder = t.nn.Sequential(
            t.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(in_channels),
            t.nn.Conv2d(in_channels, 1, 1, 1, 1),
        )
        t.nn.init.constant_(decoder[-1].weight, 0.)
        t.nn.init.constant_(decoder[-1].bias, 0.)
        return p.module(f'factor{n}', decoder, update_module_params=True)

    def model(self, x, z):
        num_genes = x['data'][0].shape[1]

        decoded = self._decode(z)

        scale = p.sample('scale', distr.Delta(
            center_crop(
                self._get_scale_decoder(decoded.shape[1]).to(decoded)(decoded),
                [None, None, *x['label'].shape[-2:]],
            )
        ))
        rim = t.cat(
            [
                self._get_factor_decoder(decoded.shape[1], n)
                .to(decoded)(decoded)
                for n in self.factors
            ],
            dim=1,
        )
        rim = center_crop(rim, [None, None, *x['label'].shape[-2:]])
        rim = t.nn.functional.softmax(rim, dim=1)
        rim = p.sample('rim', distr.Delta(rim))
        rim = scale * rim

        rmg = p.sample('rmg', distr.Delta(t.stack([
            p.sample(f'factor{n}', (
                distr.Normal(t.tensor(0.).to(z), 1.).expand([num_genes])
            ))
            for n in self.factors
        ])))

        lg = p.sample('lg', (
            distr.Normal(t.tensor(0.).to(z), 1.).expand([num_genes])
        ))
        rg = p.sample('rg', (
            distr.Normal(t.tensor(0.).to(z), 1).expand([num_genes])))

        effects = x['effects'].float()
        rgeff = p.sample('rgeff', (
            distr.Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))
        lgeff = p.sample('lgeff', (
            distr.Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))

        lg = effects @ lgeff + lg
        rg = effects @ rgeff + rg
        rmg = rg[:, None] + rmg

        with p.poutine.scale(scale=self.n/len(x)):
            with scope(prefix=self.tag):
                image = self._sample_image(x, decoded)

                def _compute_sample_params(label, rim, rmg, lg):
                    rim = t.einsum(
                        'yxi,myx->im',
                        (
                            t.eye(label.max() + 1, device=label.device)
                            [label.flatten()]
                            .reshape(*label.shape, -1)
                        ),
                        rim,
                    )
                    rgs = t.einsum('im,mg->ig', rim[1:], rmg.exp())
                    return rgs, lg.expand(len(rgs), -1)

                rgs, lg = zip(*it.starmap(
                    _compute_sample_params, zip(x['label'], rim, rmg, lg)))
                expression = p.sample(
                    'xsg',
                    distr.NegativeBinomial(
                        total_count=t.cat(rgs),
                        logits=t.cat(lg),
                    ),
                    obs=t.cat(x['data']),
                )

        return image, expression

    def guide(self, x):
        num_genes = x['data'][0].shape[1]

        for name, dim in [
            ('lg',  [num_genes]),
            ('rg',  [num_genes]),
            ('rgeff', [x['effects'].shape[1], num_genes]),
            ('lgeff', [x['effects'].shape[1], num_genes]),
            *[(f'factor{n}', [num_genes]) for n in self.factors],
        ]:
            p.sample(
                name,
                distr.Normal(
                    p.param(
                        f'{name}_mu',
                        t.zeros(dim).to(_find_device(x)),
                    ),
                    p.param(
                        f'{name}_sd',
                        1e-2 * t.ones(dim).to(_find_device(x)),
                        constraint=t.distributions.constraints.positive,
                    ),
                ),
            )

        image = super().guide(x)

        expression_encoder = p.module(
            'expression_encoder',
            t.nn.Sequential(
                t.nn.Linear(1 + num_genes, 100),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm1d(100),
                t.nn.Linear(100, 100),
            ),
            update_module_params=True,
        ).to(image)

        def encode(data, label):
            missing = t.tensor([1., *[0.] * data.shape[1]]).to(data)
            data_with_missing = t.nn.functional.pad(data, (1, 0, 1, 0))
            data_with_missing[0] = missing
            encoded_data = expression_encoder(data_with_missing)
            return t.einsum(
                'yxi,ic->cyx',
                (
                    t.eye(len(encoded_data)).to(label)
                    [label.flatten()]
                    .reshape(*label.shape, -1)
                    .float()
                ),
                encoded_data,
            )

        label = (
            t.nn.functional.interpolate(
                x['label'].float().unsqueeze(1),
                image.shape[-2:],
            )
            .squeeze(1)
            .long()
        )
        expression = t.stack([
            encode(data, label) for data, label in zip(x['data'], label)
        ])

        return t.cat([image, expression], dim=1)


class XFuse(t.nn.Module):
    def __init__(
            self,
            experiments: List[ExperimentType],
            latent_size=192,
    ):
        super().__init__()
        self.latent_size = latent_size

        self.__experiment_store = {}
        for experiment in experiments:
            self._register_experiment(experiment)

    def _get_experiment(
            self,
            experiment_type: str,
    ) -> ExperimentType:
        try:
            return self.__experiment_store[experiment_type]
        except KeyError:
            raise RuntimeError(f'unknown experiment type: {experiment_type}')

    def _register_experiment(
            self,
            experiment: ExperimentType,
    ) -> None:
        if experiment.tag in self.__experiment_store:
            raise RuntimeError(
                f'model for data type "{experiment.tag}" already registered')
        log(
            INFO,
            'registering experiment: %s (data type: "%s")',
            type(experiment).__name__,
            experiment.tag,
        )
        self.__experiment_store[experiment.tag] = experiment

    def model(self, xs):
        def _go(e, x):
            with p.poutine.scale(scale=e.n/len(x)):
                z = p.sample(f'z-{e.tag}', (
                    distr.Normal(
                        t.tensor(0., device=_find_device(x)),
                        1.,
                    )
                    .expand([1, 1, 1, 1])
                    .to_event(3)
                ))
            e.model(x, z)
            return z
        p.sample('z', distr.Delta(
            t.cat([_go(self._get_experiment(e), x) for e, x in xs.items()], 0),
        ))

    def guide(self, xs):
        def _go(e, x):
            preencoded = e.guide(x)
            z_mu = p.module(
                'z_mu',
                t.nn.Sequential(
                    t.nn.Conv2d(preencoded.shape[1], 256, 3, 1, 1),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(256),
                    t.nn.Conv2d(256, self.latent_size, 3, 1, 1),
                ),
                update_module_params=True,
            ).to(preencoded)
            z_sd = p.module(
                'z_sd',
                t.nn.Sequential(
                    t.nn.Conv2d(preencoded.shape[1], 256, 3, 1, 1),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(256),
                    t.nn.Conv2d(256, self.latent_size, 3, 1, 1),
                    t.nn.Softplus(),
                ),
                update_module_params=True,
            ).to(preencoded)
            with p.poutine.scale(scale=e.n/len(x)):
                return p.sample(f'z-{e.tag}', (
                    distr.Normal(
                        z_mu(preencoded),
                        z_sd(preencoded),
                    )
                    .to_event(3)
                ))
        p.sample('z', distr.Delta(
            t.cat([_go(self._get_experiment(e), x) for e, x in xs.items()], 0),
        ))


class Unpool(t.nn.Module):
    # TODO: move elsewhere
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=3,
            stride=2,
            padding=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if padding is None:
            padding = kernel_size // 2

        self.conv = t.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.scale_factor = stride

    def forward(self, x):
        x = t.nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(x)
        return x













def __remove_this():
    import os
    import pandas as pd
    import pyvips
    from .utility import design_matrix_from, read_data
    from .dataset import Dataset, RandomSlide, collate, spot_size

    design_file = os.path.expanduser(
        '~/histonet-test-data/mob-0.1-validation/design.small.csv')

    design = pd.read_csv(design_file)
    design_dir = os.path.dirname(design_file)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(design_dir, p)
        )

    count_data = read_data(map(_path, design.data))

    design_matrix = design_matrix_from(design[[
        x for x in design.columns
        if x not in [
                'name',
                'image',
                'labels',
                'validation',
                'data',
        ]
    ]])

    dataset = Dataset(
        [
            RandomSlide(
                data=counts,
                image=pyvips.Image.new_from_file(_path(image)),
                label=pyvips.Image.new_from_file(_path(labels)),
                patch_size=224,
            )
            for image, labels, counts in zip(
                design.image,
                design.labels,
                (count_data.loc[x] for x in count_data.index.levels[0]),
            )
        ],
        design_matrix,
    )

    loader = t.utils.data.DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=2,
        shuffle=True,
    )

    genes = list(count_data.columns)

    return (
        dataset, loader, genes,
        count_data.mean().mean() / spot_size(dataset),
        t.as_tensor(count_data.mean(0).values).log(),
    )


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


def prep(x):
    if isinstance(x, t.Tensor):
        return x.to(t.device('cuda'))
    if isinstance(x, list):
        return [prep(y) for y in x]
    if isinstance(x, dict):
        return {k: prep(v) for k, v in x.items()}


from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.path.join('/tmp/tb/', datetime.now().isoformat()))

from .logging import set_level, DEBUG
set_level(DEBUG)

from .dataset import spot_size
data, loader, genes, scale, baseline = __remove_this()
xfuse = XFuse([
    STExperiment(
        n=len(data),
        num_factors=10,
        default_scale=scale,
        gene_baseline=baseline,
    )
]).to(t.device('cuda'))
import pyro.optim
svi = p.infer.SVI(
    xfuse.model,
    xfuse.guide,
    p.optim.Adam({'lr': 1e-5}),
    p.infer.Trace_ELBO(),
)
fixed_x = prep(next(iter(loader)))


def do(i):
    results = []
    for x in loader:
        loss = svi.step(prep(x))
        writer.add_scalar('loss', loss, i)
        results.append(loss)
    print(f'{i}: {loss}')
    if i % 50 == 0:
        res = p.poutine.trace(
            p.poutine.replay(
                xfuse.model,
                p.poutine.trace(xfuse.guide).get_trace(fixed_x)
            )
        ).get_trace(fixed_x)
        rmse = (
            ((res.nodes['ST/xsg']['fn'].mean
              - res.nodes['ST/xsg']['value'])
             ** 2)
            .mean(1)
            .sqrt()
            .mean()
        )
        print(f'rmse={rmse:.2f}')
        writer.add_scalar('accuracy/rmse', rmse, i)
        for n, factor in enumerate(
                res.nodes['rim']['value'].permute(1, 0, 2, 3), 1):
            writer.add_scalar(f'activation/factor{n}', factor.mean(), i)
            writer.add_images(
                f'factors/factor{n}',
                factor.unsqueeze(1),
                i,
                dataformats='NCHW',
            )
        writer.add_scalar(
            'loss/image',
            -res.nodes['ST/image']['fn']
            .log_prob(res.nodes['ST/image']['value']).sum(),
            i,
        )
        writer.add_scalar(
            'loss/xpr',
            -res.nodes['ST/xsg']['fn']
            .log_prob(res.nodes['ST/xsg']['value']).sum(),
            i,
        )
        writer.add_images('he', (1 + res.nodes['ST/image']['value']) / 2, i)
        writer.add_images(
            'he/mean', (1 + res.nodes['ST/image']['fn'].mean) / 2, i)
        writer.add_images(
            'he/sample', (1 + res.nodes['ST/image']['fn'].sample()) / 2, i)
        writer.add_images(
            'z',
            dim_red(res.nodes['z']['value'].permute(0, 2, 3, 1)),
            i,
            dataformats='NHWC',
        )
        writer.add_images(
            'expression/activation',
            dim_red(res.nodes['rim']['value'].permute(0, 2, 3, 1)),
            i,
            dataformats='NHWC',
        )
        writer.add_images(
            'expression/scale',
            res.nodes['scale']['value'],
            i,
            dataformats='NCHW',
        )
    return results


def compare_elbo():
    guide = p.poutine.trace(xfuse.guide).get_trace(fixed_x)
    full_model = p.poutine.trace(
        p.poutine.replay(xfuse.model, guide)
    ).get_trace(fixed_x)
    reduced_model = p.poutine.trace(
        p.poutine.replay(
            p.poutine.block(lambda x: x['name'] and x['name'][:7] == 'factor0'),
            guide,
        ),
    ).get_trace(fixed_x)
