from abc import abstractmethod

from copy import deepcopy

from functools import reduce

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

from .logging import DEBUG, log
from .utility import center_crop


class ExperimentModel(t.nn.Module):
    def __init__(self, *_):
        super().__init__()

    @abstractmethod
    def forward(self, data, label, rim, rmg, lg):
        pass


class ExperimentGuide(t.nn.Module):
    def __init__(self, *_):
        super().__init__()

    @abstractmethod
    def forward(self, data, label):
        pass


class ExperimentType(NamedTuple):
    model: Type[ExperimentModel]
    guide: Type[ExperimentGuide]


__TYPE_STORE: Dict[str, ExperimentType] = {}


def _get_type(experiment_type: str) -> ExperimentType:
    try:
        return __TYPE_STORE[experiment_type]
    except KeyError:
        raise RuntimeError(f'unknown experiment type: {experiment_type}')


def _register_type(
        name: str,
        model: ExperimentModel,
        guide: ExperimentGuide
) -> None:
    if name in __TYPE_STORE:
        raise RuntimeError(
            f'model for experiment type "{name}" already registered')
    log(DEBUG, 'registering experiment type: %s', name)
    __TYPE_STORE[name] = ExperimentType(model, guide)


class DataRepresentation(t.nn.Module):
    @abstractmethod
    def model(self, x, decoded, data_plate):
        pass

    @abstractmethod
    def guide_pre(self, x, data_plate) -> t.Tensor:
        pass

    @abstractmethod
    def guide_post(self, x, pre, decoded, data_plate) -> None:
        pass


class Image(DataRepresentation):
    def model(self, x, decoded, data_plate):
        nc = decoded.shape[1]
        img_mu = p.module(
            'img_mu',
            t.nn.Sequential(
                t.nn.Conv2d(nc, nc, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(nc),
                t.nn.Conv2d(nc, 3, 3, 1, 1),
                t.nn.Tanh(),
            ),
            update_module_params=True,
        ).to(decoded)
        img_sd = p.module(
            'img_sd',
            t.nn.Sequential(
                t.nn.Conv2d(nc, nc, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(nc),
                t.nn.Conv2d(nc, 3, 3, 1, 1),
                t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)
        mu = img_mu(center_crop(decoded, [None, None, *x['image'].shape[-2:]]))
        sd = img_sd(center_crop(decoded, [None, None, *x['image'].shape[-2:]]))
        with data_plate:
            p.sample(
                'image',
                distr.Normal(mu, sd).to_event(3),
                obs=x['image'],
            )
        # img_sd = p.module(
        #     'img_sd',
        #     t.nn.Sequential(
        #         t.nn.Conv2d(nc, nc, 3, 1, 1),
        #         t.nn.LeakyReLU(0.2, inplace=True),
        #         t.nn.BatchNorm2d(nc),
        #         t.nn.Conv2d(nc, 3 * 3, 3, 1, 1),
        #         t.nn.Softplus(),
        #     ),
        #     update_module_params=True,
        # ).to(decoded)
        # mu = center_crop(img_mu(decoded), [None, None, *x['image'].shape[-2:]])
        # sd = center_crop(img_sd(decoded), [None, None, *x['image'].shape[-2:]])
        # with data_plate:
        #     p.sample(
        #         'image',
        #         (
        #             distr.MultivariateNormal(
        #                 mu.permute(0, 2, 3, 1).reshape(-1, 3),
        #                 sd.permute(0, 2, 3, 1).reshape(-1, 3, 3),
        #             )
        #             .reshape(mu.shape[0], *mu.shape[2:], 3)
        #             .permute(0, 3, 1, 2)
        #             .to_event(3)
        #         ),
        #         obs=x['image'],
        #     )
        return mu

    def guide_pre(self, x, data_plate):
        return x['image']

    def guide_post(self, x, pre, decoded, data_plate):
        pass


class GeneExpression(DataRepresentation):
    def __init__(
            self,
            genes: List[str],
            gene_baseline: Optional[np.ndarray] = None,
            covariates: Optional[List[Tuple[str, List[str]]]] = None,
            truncation_threshold: int = 20,
    ):
        super().__init__()

        self.genes = list(genes)
        self.covariates = list(covariates or [])
        self.covariates_size = reduce(
            op.add, map(lambda x: len(x[1]), self.covariates), 0)
        self.truncation_threshold = truncation_threshold

    def model(self, x, decoded, data_plate):
        nc = decoded.shape[1]

        with data_plate:
            decoder_activation = p.module(
                'decoder_activation',
                t.nn.Sequential(
                    t.nn.Conv2d(nc, nc, 5, padding=2),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(nc),
                ),
                update_module_params=True,
            ).to(decoded)
            decoder_activation_a = p.module(
                'decoder_activation_a',
                t.nn.Sequential(
                    t.nn.Conv2d(nc, self.truncation_threshold, 5, padding=2),
                ),
                update_module_params=True,
            ).to(decoded)
            decoder_activation_b = p.module(
                'decoder_activation_b',
                t.nn.Sequential(
                    t.nn.Conv2d(nc, self.truncation_threshold, 5, padding=2),
                ),
                update_module_params=True,
            ).to(decoded)

            activation_state = decoder_activation(decoded)
            activation_a = (
                center_crop(
                    decoder_activation_a(activation_state),
                    [None, None, *x['labels'].shape[-2:]],
                )
                .permute(0, 2, 3, 1)
            )
            activation_b = (
                center_crop(
                    decoder_activation_b(activation_state),
                    [None, None, *x['labels'].shape[-2:]],
                )
                .permute(0, 2, 3, 1)
            )
            # https://github.com/pytorch/pytorch/pull/20244
            # activation = p.sample(
            #     'activation_coeffs',
            #     distr.Beta(activation_a, activation_b).to_event(1),
            # )
            activation = p.sample(
                'activation_coeffs',
                distr.Normal(
                    activation_a,
                    t.nn.functional.softplus(activation_b),
                ).to_event(3),
            )
            activation = t.sigmoid(activation)
            activation_accumulated = t.stack(
                [
                    t.ones(*activation.shape[:-1], device=activation.device),
                    *(
                        (1 - activation[..., :i]).prod(-1)
                        for i in range(1, self.truncation_threshold)
                    ),
                ],
                -1,
            )
            activation = activation * activation_accumulated

            decoder_scaling = p.module(
                'decoder_scaling',
                t.nn.Sequential(
                    t.nn.Conv2d(nc, nc, 5, padding=2),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(nc),
                    t.nn.Conv2d(nc, 1, 5, padding=2),
                    t.nn.Softplus(),
                ),
                update_module_params=True,
            ).to(decoded)
            scaling = p.sample(
                'scaling',
                distr.Delta(
                    center_crop(
                        decoder_scaling(decoded).permute(0, 2, 3, 1),
                        activation.shape[:3],
                    ),
                ).to_event(3),
            )

            rim = p.sample(
                'rim',
                distr.Delta(scaling * activation).to_event(3),
            )

        rmg = p.sample('rmg', (
            distr.Normal(
                t.tensor(0., device=x['data'][0].device),
                1.,
            )
            .expand([self.truncation_threshold, len(self.genes)])
            .to_event(2)
        ))

        lg = p.sample('lg', (
            distr.Normal(
                t.tensor(0., device=x['data'][0].device),
                1.,
            )
            .expand([len(self.genes)])
            .to_event(1)
        ))

        effects = t.cat(
            [
                t.ones(
                    x['effects'].shape[0],
                    1,
                    dtype=t.float32,
                    device=x['effects'].device,
                ),
                x['effects'].float(),
            ],
            1,
        )
        rgeff = p.sample('rgeff', (
            distr.Normal(
                t.tensor(0., device=x['data'][0].device),
                1.
            )
            .expand([effects.shape[1], len(self.genes)])
            .to_event(1)
        ))
        lgeff = p.sample('lgeff', (
            distr.Normal(
                t.tensor(0., device=x['data'][0].device),
                1.
            )
            .expand([effects.shape[1], len(self.genes)])
            .to_event(1)
        ))

        rmg = (effects @ rgeff)[:, None] + rmg
        lg = effects @ lgeff + lg

        for i, (type, data, label, rim_, rmg_, lg_) in enumerate(
                zip(x['type'], x['data'], x['labels'], rim, rmg, lg)):
            type_model = p.module(
                f'{type}_model',
                _get_type(type).model(data, label, rim_, rmg_, lg_),
                update_module_params=True,
            ).train(self.training)
            with data_plate, scope(prefix=f'sample{i}'):
                type_model(data, label, rim_, rmg_, lg_)

        return rim, rmg, lg

    def _sample_globals(self, x):
        device = x['data'][0].device

        globals_list = [
            ('rmg', [self.truncation_threshold, len(self.genes)]),
            ('lg',  [len(self.genes)]),
            ('rgeff', [1 + x['effects'].shape[1], len(self.genes)]),
            ('lgeff', [1 + x['effects'].shape[1], len(self.genes)]),
        ]

        nglobals = sum(reduce(op.mul, dims) for _name, dims in globals_list)
        # TODO: inject dependency
        globals_sample = p.sample('globals', (
            distr.Normal(
                p.param('globals_mu', t.zeros(nglobals)).to(device),
                t.nn.functional.softplus(
                    p.param('globals_sd', t.zeros(nglobals)).to(device)
                ),
            )
            .to_event(1)
        ))

        remaining = globals_sample.clone()

        # distribute sampled variables to the sites used in the model
        for name, dims in globals_list:
            n = reduce(op.mul, dims)
            p.sample(name, distr.Delta(remaining[:n].reshape(dims)))
            remaining = remaining[n:]

        assert len(remaining) == 0

        return globals_sample

    def _get_spatial_encoding(self, x):
        encoded: List[t.Tensor] = []

        for i, (type, data, label) in enumerate(
                zip(x['type'], x['data'], x['labels'])):
            type_guide = p.module(
                f'{type}_guide',
                _get_type(type).guide(data, label),
                update_module_params=True,
            ).train(self.training)
            with scope(prefix=f'sample{i}'):
                precoded = type_guide(data, label)
            encode = p.module(
                f'{type}_encode',
                t.nn.Conv2d(precoded.shape[-1], 10, 1, bias=False),
                update_module_params=True,
            ).to(precoded)
            encoded.append(encode(
                precoded
                .permute(2, 0, 1)
                .unsqueeze(0)
            ))

        encoded = t.cat(encoded)

        return encoded

    def _sample_activation(self, decoded):
        guide_activation = p.module(
            'guide_activation',
            t.nn.Sequential(
                t.nn.Conv2d(decoded.shape[1], 10, 5, padding=2),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(10),
            ),
            update_module_params=True,
        ).to(decoded)
        guide_activation_a = p.module(
            'guide_activation_a',
            t.nn.Sequential(
                t.nn.Conv2d(10, self.truncation_threshold, 5, padding=2),
                # t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)
        guide_activation_b = p.module(
            'guide_activation_b',
            t.nn.Sequential(
                t.nn.Conv2d(10, self.truncation_threshold, 5, padding=2),
                # t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)

        activation_state = guide_activation(decoded)
        activation_a = (
            guide_activation_a(activation_state)
            .permute(0, 2, 3, 1)
        )
        activation_b = (
            guide_activation_b(activation_state)
            .permute(0, 2, 3, 1)
        )
        # activation = p.sample(
        #     'activation_coeffs',
        #     distr.Beta(activation_a, activation_b).to_event(1),
        # )
        activation = p.sample(
            'activation_coeffs',
            distr.Normal(
                activation_a,
                t.nn.functional.softplus(activation_b),
            ).to_event(3),
        )

        return activation.permute(0, 3, 1, 2)

    def guide_pre(self, x, data_plate):
        globals = self._sample_globals(x)
        global_encoding = p.module(
            'global_encoding',
            t.nn.Sequential(
                t.nn.Linear(len(globals), 3),
            ),
            update_module_params=True,
        ).to(globals)(globals)
        spatial_encoding = self._get_spatial_encoding(x)
        return t.cat(
            [
                # spatial_encoding,
                (
                    global_encoding
                    .expand(
                        spatial_encoding.shape[0],
                        *spatial_encoding.shape[2:],
                        -1,
                    )
                    .permute(0, 3, 1, 2)
                ),
                (
                    x['effects']
                    .expand(*spatial_encoding.shape[2:], -1, -1)
                    .permute(2, 3, 0, 1)
                    .float()
                ),
            ],
            1,
        )

    def guide_post(self, x, pre, decoded, data_plate):
        decoded = center_crop(decoded, [None, None, *pre.shape[-2:]])
        with data_plate:
            self._sample_activation(t.cat([pre, decoded], 1))


class Combined(Image, GeneExpression):
    def __init__(self, *args, **kwargs):
        Image.__init__(self)
        GeneExpression.__init__(self, *args, **kwargs)

    def model(self, x, decoded, data_plate):
        return [
            Image.model(self, x, decoded, data_plate),
            GeneExpression.model(self, x, decoded, data_plate),
        ]

    def guide_pre(self, x, data_plate):
        encoded_image = Image.guide_pre(self, x, data_plate)
        encoded_xpr = GeneExpression.guide_pre(self, x, data_plate)
        return t.cat([encoded_image, encoded_xpr], 1)

    def guide_post(self, x, pre, decoded, data_plate):
        Image.guide_post(self, x, pre, decoded, data_plate)
        GeneExpression.guide_post(self, x, pre, decoded, data_plate)


class XFuse(t.nn.Module):
    def __init__(
            self,
            representation,
            latent_size=192,
            feature_size=32,
    ):
        super().__init__()

        self.representation = representation
        self.add_module('_representation', self.representation)

        self.latent_size = latent_size

        self.feature_size = feature_size

        self._decoder = t.nn.Sequential(
            t.nn.Conv2d(
                self.latent_size, 16 * self.feature_size, 5, padding=5),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * self.feature_size),
            Unpool(16 * self.feature_size, 8 * self.feature_size, 5),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * self.feature_size),
            Unpool(8 * self.feature_size, 4 * self.feature_size, 5),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * self.feature_size),
            Unpool(4 * self.feature_size, 2 * self.feature_size, 5),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * self.feature_size),
            Unpool(2 * self.feature_size, self.feature_size, 5),
            # x1
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(self.feature_size),
        )

        self._guide_decoder = deepcopy(self._decoder)

        self.latent_loc = t.nn.Sequential(
            t.nn.Conv2d(
                16 * self.feature_size, 16 * self.feature_size, 3, 1, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * self.feature_size),
            t.nn.Conv2d(16 * self.feature_size, latent_size, 3, 1, 1),
        )
        self.latent_scale = t.nn.Sequential(
            t.nn.Conv2d(
                16 * self.feature_size, 16 * self.feature_size, 3, 1, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * self.feature_size),
            t.nn.Conv2d(16 * self.feature_size, latent_size, 3, 1, 1),
            t.nn.Softplus(),
        )

    def _get_decoder(self, x):
        return p.module('decoder', self._decoder)

    def _get_guide_decoder(self, x):
        return p.module('guide_decoder', self._guide_decoder)

    def _get_encoder(self, x):
        return p.module(
            'encoder',
            t.nn.Sequential(
                # x1
                t.nn.Conv2d(x.shape[1], 2 * self.feature_size, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(2 * self.feature_size),
                # x2
                t.nn.Conv2d(
                    2 * self.feature_size, 4 * self.feature_size, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(4 * self.feature_size),
                # x4
                t.nn.Conv2d(
                    4 * self.feature_size, 8 * self.feature_size, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(8 * self.feature_size),
                # x8
                t.nn.Conv2d(
                    8 * self.feature_size, 16 * self.feature_size, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16 * self.feature_size),
                # x16
            ),
            update_module_params=True,
        ).to(x)

    def model(self, x):
        data_plate = p.plate(
            'data',
            size=x['dataset_size'],
            subsample_size=x['batch_size'],
        )
        with data_plate:
            z = p.sample(
                'z',
                (
                    distr.Normal(
                        t.tensor(0., device=next(self.parameters()).device),
                        1.,
                    )
                    .expand([1, 1, 1, 1])
                    .to_event(3)
                ),
            )
        decoded = self._get_decoder(z)(z)
        return self.representation.model(x, decoded, data_plate)

    def guide(self, x):
        data_plate = p.plate(
            'data',
            size=x['dataset_size'],
            subsample_size=x['batch_size'],
        )
        pre = self.representation.guide_pre(x, data_plate)
        encoded = self._get_encoder(pre)(pre)
        loc = p.module('latent_loc', self.latent_loc)(encoded)
        scale = p.module('latent_scale', self.latent_scale)(encoded)
        with data_plate:
            z = p.sample('z', distr.Normal(loc, scale).to_event(3))
        decoded = self._get_guide_decoder(z)(z)
        self.representation.guide_post(x, pre, decoded, data_plate)


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


class STModel(ExperimentModel):
    def forward(self, data, label, rim, rmg, lg):
        rim = t.einsum(
            'yxi,yxf->if',
            (
                t.eye(label.max() + 1, device=label.device)
                [label.flatten()]
                .reshape(*label.shape, -1)
            ),
            rim,
        )
        rgs = t.einsum('im,mg->ig', rim[1:], rmg.exp())
        xgs = p.sample(
            'xgs',
            (
                distr.NegativeBinomial(
                    total_count=rgs,
                    logits=lg,
                )
                .to_event(2)
            ),
            obs=data,
        )
        return xgs


class STGuide(ExperimentGuide):
    def __init__(self, data, label):
        super().__init__(data, label)
        self.xpr_encoder = t.nn.Linear(1 + data.shape[1], 10).to(data)

    def forward(self, data, label):
        data_with_missing = t.nn.functional.pad(data, (1, 0, 1, 0))
        data_with_missing[0, 0] = 1.

        if self.training:
            data_with_missing[
                t.distributions.Bernoulli(0.5)
                .sample((len(data_with_missing), ))
                .byte()
            ] = t.tensor([1., *[0.] * data.shape[1]], device=data.device)

        convolved_data = self.xpr_encoder(data_with_missing)

        return t.einsum(
            'yxi,ic->yxc',
            (
                t.eye(len(convolved_data))
                .to(label)
                [label.flatten()]
                .reshape(*label.shape, -1)
                .float()
            ),
            convolved_data,
        )


_register_type('ST', STModel, STGuide)













def __remove_this():
    import os
    import pandas as pd
    import pyvips
    from .utility import design_matrix_from, read_data
    from .dataset import Dataset, RandomSlide, collate

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

    return dataset, loader, genes


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
    x = {
        k: v.to(t.device('cuda')) if isinstance(v, t.Tensor) else v
        for k, v in x.items()
    }
    x['data'] = [v.to(t.device('cuda')) for v in x['data']]
    return x


from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.path.join('/tmp/tb/', datetime.now().isoformat()))

data, loader, genes = __remove_this()
xfuse = XFuse(Combined(genes)).to(t.device('cuda'))
import pyro.optim
svi = p.infer.SVI(
    xfuse.model,
    xfuse.guide,
    p.optim.Adam({'lr': 1e-5}),
    p.infer.Trace_ELBO(),
)
fixed_x = prep(next(iter(loader)))


def do(i):
    print(i)
    results = []
    for x in loader:
        loss = svi.step(prep(x))
        writer.add_scalar('loss', loss, i)
        results.append(loss)
    if i % 50 == 0:
        res = p.poutine.trace(
            p.poutine.replay(
                xfuse.model,
                p.poutine.trace(xfuse.guide).get_trace(fixed_x)
            )
        ).get_trace(fixed_x)
        writer.add_scalar(
            'accuracy/rmse',
            (
                ((res.nodes['sample0/xgs']['fn'].mean
                  - res.nodes['sample0/xgs']['value'])
                 ** 2)
                .mean(1)
                .sqrt()
                .mean()
            ),
            i,
        )
        for i, factor in enumerate(
                res.nodes['rim']['value'].permute(3, 0, 1, 2), 1):
            writer.add_scalar(f'activation/factor{i}', factor.mean(), i)
        writer.add_scalar(
            'loss/image',
            -res.nodes['image']['fn']
            .log_prob(res.nodes['image']['value']).sum(),
            i,
        )
        writer.add_scalar(
            'loss/xpr',
            -res.nodes['sample0/xgs']['fn']
            .log_prob(res.nodes['sample0/xgs']['value']).sum(),
            i,
        )
        writer.add_images('he', (1 + res.nodes['image']['value']) / 2, i)
        writer.add_images(
            'he/mean', (1 + res.nodes['image']['fn'].mean) / 2, i)
        writer.add_images(
            'he/sample', (1 + res.nodes['image']['fn'].sample()) / 2, i)
        writer.add_images(
            'z',
            dim_red(res.nodes['z']['value'].permute(0, 2, 3, 1)),
            i,
            dataformats='NHWC',
        )
        writer.add_images(
            'activation',
            dim_red(res.nodes['rim']['value']),
            i,
            dataformats='NHWC',
        )
        writer.add_images(
            'activation/coeffs',
            dim_red(res.nodes['activation_coeffs']['fn'].mean),
            i,
            dataformats='NHWC',
        )
    return results
