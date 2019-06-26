from abc import abstractmethod

from copy import deepcopy

import itertools as it

from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np

import pyro as p
from pyro.contrib.autoname import scope
import pyro.distributions as distr

import torch as t

from .logging import DEBUG, INFO, log
from .utility import (
    Unpool,
    center_crop,
    find_device,
    sparseonehot,
)


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
            factors: List[Tuple[float, t.Tensor]] = [],
            default_scale: float = 1.,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__factors: Dict[str, Tuple(float, t.Tensor)] = {}
        self.__factors_counter = it.count()
        for factor in factors:
            self.add_factor(factor)

        self.__default_scale = default_scale

    @property
    def factors(self):
        return deepcopy(self.__factors)

    def add_factor(self, factor=None):
        if factor is None:
            factor = (0., None)
        n = next(self.__factors_counter)
        assert n not in self.__factors
        log(DEBUG, 'adding new factor: %d', n)
        self.__factors.setdefault(n, factor)
        return self

    def remove_factor(self, n):
        log(DEBUG, 'removing factor: %d', n)
        try:
            self.__factors.pop(n)
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
        t.nn.init.constant_(decoder[-1].bias, self.__factors[n][0])
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

        effects = x['effects'].float()
        rgeff = p.sample('rgeff', (
            distr.Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))
        lgeff = p.sample('lgeff', (
            distr.Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))

        lg = effects @ lgeff
        rg = effects @ rgeff
        rmg = rg[:, None] + rmg

        with p.poutine.scale(scale=self.n/len(x)):
            with scope(prefix=self.tag):
                image = self._sample_image(x, decoded)

                def _compute_sample_params(label, rim, rmg, lg):
                    labelonehot = sparseonehot(label.flatten())
                    rim = t.sparse.mm(
                        labelonehot.t().float(),
                        rim.permute(1, 2, 0).view(-1, rim.shape[0]),
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
            ('rgeff', [x['effects'].shape[1], num_genes]),
            ('lgeff', [x['effects'].shape[1], num_genes]),
        ]:
            p.sample(
                name,
                distr.Normal(
                    p.param(
                        f'{name}_mu',
                        t.zeros(dim),
                    ).to(find_device(x)),
                    p.param(
                        f'{name}_sd',
                        1e-2 * t.ones(dim),
                        constraint=t.distributions.constraints.positive,
                    ).to(find_device(x)),
                ),
            )

        for n, (_, factor_default) in self.factors.items():
            if factor_default is None:
                factor_default = t.zeros(num_genes)
            p.sample(
                f'factor{n}',
                distr.Normal(
                    p.param(
                        f'factor{n}_mu',
                        factor_default.float(),
                    ).to(find_device(x)),
                    p.param(
                        f'factor{n}_sd',
                        1e-2 * t.ones_like(factor_default).float(),
                        constraint=t.distributions.constraints.positive,
                    ).to(find_device(x)),
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
            labelonehot = sparseonehot(label.flatten(), len(encoded_data))
            expanded = t.sparse.mm(labelonehot.float(), encoded_data)
            return expanded.t().reshape(-1, *label.shape)

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
                        t.tensor(0., device=find_device(x)),
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
        count_data,
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
data, loader, genes, scale, baseline, counts = __remove_this()
xfuse = XFuse([
    STExperiment(
        n=len(data),
        default_scale=scale,
        factors=[
            (0., baseline),
            (-10, None),
        ],
    )
]).to(t.device('cuda'))
import pyro.optim
svi = p.infer.SVI(
    xfuse.model,
    xfuse.guide,
    p.optim.Adam({'lr': 1e-3}),
    p.infer.Trace_ELBO(),
)
fixed_x = prep(next(iter(loader)))


def normalize(img):
    mins = img.permute(0, 2, 3, 1).reshape(-1, img.shape[1]).min(0).values[None, :, None, None]
    maxs = img.permute(0, 2, 3, 1).reshape(-1, img.shape[1]).max(0).values[None, :, None, None]
    return (img - mins) / (maxs - mins)


def do(i):
    results = []
    for x in loader:
        loss = svi.step(prep(x))
        writer.add_scalar('loss', loss, i)
        results.append(loss)
    print(f'{i}: {loss}')
    if i % 100 == 0:
        print('starting factor purge')
        with t.no_grad():
            def _model_without(n):
                reduced_model = deepcopy(xfuse)
                reduced_model._XFuse__experiment_store['ST'].remove_factor(n)
                return reduced_model

            reduced_models, ns = zip(*[
                (_model_without(n), n)
                for n in xfuse._get_experiment('ST').factors
            ])

            def _compare_once():
                guide = p.poutine.trace(xfuse.guide).get_trace(fixed_x)

                def _evaluate(model):
                    return (
                        p.poutine.trace(p.poutine.replay(model, guide))
                        .get_trace(fixed_x)
                        .log_prob_sum()
                        .item()
                    )

                full = _evaluate(xfuse.model)
                deltas = [
                    _evaluate(xfuse.model) - full for xfuse in reduced_models]
                return deltas

            res = [_compare_once() for _ in range(10)]
            res = np.array(res).mean(0)
            dubious = [
                n for res, n in reversed(sorted(zip(res, ns)))
                if res >= 0
            ]
            if dubious == []:
                print('no factors are dubious')
                xfuse._get_experiment('ST').add_factor((-10., None))
            else:
                print(
                    'the following factors are dubious: '
                    + ', '.join(map(str, dubious))
                )
                for n in dubious[:-1][:len(res) - 2]:
                    xfuse._get_experiment('ST').remove_factor(n)
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
        for n, factor in zip(
                xfuse._get_experiment('ST').factors.keys(),
                res.nodes['rim']['value'].permute(1, 0, 2, 3),
        ):
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
        if res.nodes['rim']['value'].shape[1] >= 3:
            writer.add_images(
                'expression/activation',
                dim_red(res.nodes['rim']['value'].permute(0, 2, 3, 1)),
                i,
                dataformats='NHWC',
            )
        writer.add_images(
            'expression/scale',
            normalize(res.nodes['scale']['value']),
            i,
            dataformats='NCHW',
        )
    return results
