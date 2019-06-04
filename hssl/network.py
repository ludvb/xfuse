from abc import abstractmethod

from functools import reduce

import itertools as it

import operator as op

from typing import List, Optional, Tuple

import numpy as np

import torch as t

from .distributions import (
    Distribution,
    Beta,
    Normal,
    Variable,
    kl_divergence,
)
from .logging import DEBUG, log
from .utility import center_crop
from .utility.init_args import store_init_args


class Variational(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latents = []

    def _register_latent(
            self,
            variable: Variable,
            prior: Distribution,
            id: str,
            is_global: bool = False,
    ):
        if id in self._latents:
            raise RuntimeError(f'variable {id} has already been registered')

        log(DEBUG, 'registering latent variable %s', id)
        setattr(self, f'{id}', variable)
        setattr(self, f'{id}_p', prior)
        setattr(self, f'{id}_is_global', is_global)
        self._latents.append(id)

    def _get_latent(self, latent_id: str):
        return (
            getattr(self, f'{latent_id}'),
            getattr(self, f'{latent_id}_p'),
            getattr(self, f'{latent_id}_is_global'),
        )

    def _make_covariate(
            self,
            name,
            shape,
            pdistr=Normal,
            pparams={'loc': 0., 'scale': np.log(np.exp(1) - 1)},
            qdistr=Normal,
            qparams={'loc': 0., 'scale': -1e5},
    ):
        def _register_parameter(suffix, param):
            self.register_parameter(f'{name}_{suffix}', param)
            return param

        v = Variable(qdistr().set(
            **{
                k: _register_parameter(
                    f'q_{k}', t.nn.Parameter(v * t.ones(shape)))
                for k, v in qparams.items()
            },
            r_transform=True,
        ))

        p = pdistr().set(
            **{
                k: _register_parameter(
                    f'p_{k}',
                    t.nn.Parameter(t.as_tensor(v), requires_grad=False),
                )
                for k, v in pparams.items()
            },
            r_transform=True,
        )

        self._register_latent(v, p, name, True)

    def add_module(self, name, module):
        if isinstance(module, Variational):
            for id, variable, prior, is_global in [
                    (x, *module._get_latent(x)) for x in module._latents]:
                self._register_latent(variable, prior, id, is_global)
        return super().add_module(name, module)

    def resample_globals(self):
        for v in (v for v, _p, g in map(self._get_latent, self._latents) if g):
            v.sample()
        return self

    def complexity_cost(self, batch_fraction):
        return sum([
            kl_divergence(x, p) * (batch_fraction if g else 1.)
            for x, p, g in map(self._get_latent, self._latents)
        ])


class Unpool(t.nn.Module):
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


class Encoder(t.nn.Module):
    @abstractmethod
    def output_channels(self):
        pass


class Decoder(t.nn.Module):
    def statistics(self, x, result):
        return {}

    @abstractmethod
    def loglikelihood(self, x, result):
        pass


class STEncoder(Encoder):
    def __init__(
            self,
            genes: List[str],
            output_size: int = 100,
    ):
        super().__init__()
        self.genes = list(genes)
        self._output_channels = output_size
        self.xpr_encoder = t.nn.Linear(1 + len(self.genes), output_size)

    def output_channels(self):
        return self._output_channels

    def forward(self, x):
        label, data = x['label'], x['data']

        data_with_missing = t.nn.functional.pad(data, (1, 0, 1, 0))
        data_with_missing[0, 0] = 1.

        # if self.training:
        #     data_with_missing[
        #         t.distributions.Bernoulli(0.5)
        #         .sample((len(data_with_missing), ))
        #         .byte()
        #     ] = t.tensor([1., *[0.] * data.shape[1]], device=data.device)

        convolved_data = self.xpr_encoder(data_with_missing)
        return t.einsum(
            'byxi,ic->bcyx',
            (
                t.eye(len(convolved_data))
                .to(label)
                [label.flatten()]
                .reshape(*label.shape, -1)
                .float()
            ),
            convolved_data,
        )


class STDecoder(Decoder, Variational):
    def __init__(
            self,
            genes: List[str],
            gene_baseline: Optional[np.ndarray] = None,
            covariates: Optional[List[Tuple[str, List[str]]]] = None,
            truncation_threshold: int = 10,
            feature_size: int = None
    ):
        super().__init__()

        self.genes = list(genes)

        self._make_covariate('lg', (len(genes), ))
        self._make_covariate('rg', (len(genes), ))

        self._make_covariate('rgp', (truncation_threshold, len(genes)))
        self._make_covariate(
            'r',
            (self.factors, ),
            pdistr=Beta, pparams={
                'shape1': np.log(np.exp(1) - 1),
                'shape2': np.log(np.exp(1) - 1),
            },
            qdistr=Beta, qparams={
                'shape1': np.log(np.exp(1) - 1),
                'shape2': np.log(np.exp(1) - 1),
            },
        )

        self.gene_decoder = t.nn.Conv2d(
            feature_size, len(genes), 11, 1, 5, bias=False)

        if covariates is not None and len(covariates) > 0:
            self._covariates = covariates
            n_fe = reduce(op.add, map(lambda x: len(x[1]), covariates))
            self._make_covariate('rgeff', (n_fe, len(genes)))
            self._make_covariate('lgeff', (n_fe, len(genes)))
        else:
            self._covariates = []

        if gene_baseline is not None:
            if len(gene_baseline) != len(genes):
                raise ValueError(
                    'size of `gene_baseline` does not match `genes`'
                    f' ({gene_baseline.shape[1]} vs. {len(genes)})'
                )
            lgb = t.tensor(np.log(gene_baseline)).float()
            self.rg_q_loc.data = lgb.clone()
            self.rg_p_loc.data = lgb.clone()

    def forward(self, x, decoded):
        idxs = np.random.choice(len(self.genes), 100, False)

        rate = t.nn.functional.conv2d(
            input=decoded,
            weight=self.gene_decoder.weight[idxs],
            padding=self.gene_decoder.padding,
        )

        rate = t.einsum(
            'byxi,bgyx->ig',
            (
                t.eye(
                    x['label'].max() + 1,
                    device=x['label'].device,
                )
                [x['label']]
            ),
            rate.exp(),
        )

        rate = rate * self.rg.value[idxs].unsqueeze(0)
        logits = self.lg.value[idxs].unsqueeze(0)

        if x['effects'] is not None:
            effects = x['effects'].float()
            rate = rate * (effects @ self.rgeff.value[idxs])
            logits = logits + (effects @ self.lgeff.value[idxs])

        rate_ = t.zeros((len(rate), len(self.genes))).fill_(float('nan'))
        rate_[:, idxs] = rate
        logits_ = t.zeros((len(rate), len(self.genes))).fill_(float('nan'))
        logits_[:, idxs] = logits

        return rate_, logits_

    def statistics(self, x, result):
        rate, logits = result
        d = t.distributions.NegativeBinomial(
            total_count=rate,
            logits=logits,
        )
        return {
            'rmse': t.mean(
                t.sqrt(t.mean((d.mean - x['data']) ** 2, 1))
                .masked_select(x['data'].sum(1) != 0)
            ),
        }

    def loglikelihood(self, x, result):
        rate, logits = result
        return (
            t.distributions.NegativeBinomial(
                total_count=rate,
                logits=logits,
            )
            .log_prob(x['data']).sum()
        )


class HEEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def output_channels(self):
        return 3

    def forward(self, x):
        return x['image']


class HEDecoder(Decoder):
    def __init__(self, feature_size: int = None):
        super().__init__()

        self.img_mu = t.nn.Sequential(
            t.nn.Conv2d(feature_size, feature_size, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(feature_size),
            t.nn.Conv2d(feature_size, 3, 3, 1, 1, bias=True),
            t.nn.Tanh(),
        )
        self.img_sd = t.nn.Sequential(
            t.nn.Conv2d(feature_size, feature_size, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(feature_size),
            t.nn.Conv2d(feature_size, 3, 3, 1, 1, bias=True),
            t.nn.Softplus(),
        )

    def forward(self, x, decoded):
        img_mu = self.img_mu(decoded)
        img_sd = self.img_sd(decoded)
        return img_mu, img_sd

    def loglikelihood(self, x, result):
        img_mu, img_sd = result
        return (
            t.distributions.Normal(img_mu, img_sd)
            .log_prob(x['image']).sum()
        )


@store_init_args
class XFuse(Variational):
    def __init__(
            self,
            encoders,
            decoders,
            latent_size=192,
            feature_size=32,
            dataset_size=float('inf'),
    ):
        super().__init__()

        self.dataset_size = dataset_size

        def _register_module(module):
            name = type(module).__name__
            log(DEBUG, 'registering sub-module %s', name)
            self.add_module(name, module)
            return name, module

        self.encoders = {
            k: v for k, v in [_register_module(x()) for x in encoders]
        }
        self.decoders = {
            k: v for k, v in
            (_register_module(x(feature_size=feature_size)) for x in decoders)
        }

        input_size = reduce(
            op.add,
            (x.output_channels() for x in self.encoders.values()),
        )

        self.encoder = t.nn.Sequential(
            # x1
            t.nn.Conv2d(input_size, 2 * feature_size, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * feature_size),
            # x2
            t.nn.Conv2d(2 * feature_size, 4 * feature_size, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * feature_size),
            # x4
            t.nn.Conv2d(4 * feature_size, 8 * feature_size, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * feature_size),
            # x8
            t.nn.Conv2d(8 * feature_size, 16 * feature_size, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * feature_size),
            # x16
        )

        self.z_mu = t.nn.Conv2d(16 * feature_size, latent_size, 3, 1, 1)
        self.z_sd = t.nn.Conv2d(16 * feature_size, latent_size, 3, 1, 1)
        self.z = Variable(Normal())
        self._register_latent(self.z, Normal(), 'z')

        self.decoder = t.nn.Sequential(
            t.nn.Conv2d(latent_size, 16 * feature_size, 5, padding=4),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * feature_size),
            Unpool(16 * feature_size, 8 * feature_size, 5),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * feature_size),
            Unpool(8 * feature_size, 4 * feature_size, 5),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * feature_size),
            Unpool(4 * feature_size, 2 * feature_size, 5),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * feature_size),
            Unpool(2 * feature_size, feature_size, 5),
            # x1
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(feature_size),
        )

    def forward(self, x, compute_loss=True, collect_stats=True):
        input_volume = t.cat([f(x) for f in self.encoders.values()], 1)
        encoded = self.encoder(input_volume)

        z_mu = self.z_mu(encoded)
        z_sd = self.z_sd(encoded)

        self.z.distribution.set(
            loc=z_mu,
            scale=z_sd,
            r_transform=True,
        )
        z = self.z.sample().value

        decoded = center_crop(
            self.decoder(z), [None, None, *input_volume.shape[-2:]])

        results = {n: f(x, decoded) for n, f in self.decoders.items()}

        elbo = (
            reduce(
                op.add,
                (
                    f.loglikelihood(x, r) for f, r in
                    zip(self.decoders.values(), results.values())
                ),
            )
            -
            self.complexity_cost(len(x) / self.dataset_size)
            if compute_loss else
            t.tensor(0., device=z.device)
        )

        stats = (
            {
                k: v.unsqueeze(0)
                for f, r in zip(self.decoders.values(), results.values())
                for k, v in f.statistics(x, r).items()
            }
            if collect_stats else
            {}
        )

        return dict(
            loss=(-elbo).unsqueeze(0),
            stats=stats,
            **results,
        )


@store_init_args
class Histonet(Variational):
    def __init__(
            self,
            genes,
            latent_size=192,
            nf=32,
    ):
        super().__init__()

        self.genes = genes

        self.img_encoder = t.nn.Sequential(
            # x1
            t.nn.Conv2d(3, 2 * nf, 4, 2, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * nf),
            # x2
            t.nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * nf),
            # x4
            t.nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * nf),
            # x8
            t.nn.Conv2d(8 * nf, 16 * nf, 4, 2, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            # x16
        )
        self.xpr_encoder1 = t.nn.Linear(1 + len(self.genes), 100)
        self.xpr_encoder2 = t.nn.Sequential(
            t.nn.Conv2d(100, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
        )

        self.z_mu = t.nn.Sequential(
            t.nn.Conv2d(32 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, latent_size, 3, 1, 1, bias=True),
        )
        self.z_sd = t.nn.Sequential(
            t.nn.Conv2d(32 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, latent_size, 3, 1, 1, bias=True),
        )
        self.z = Variable(Normal())
        self._register_latent(self.z, Normal(), 'z')

        self.decoder = t.nn.Sequential(
            t.nn.Conv2d(latent_size, 16 * nf, 5, padding=4),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            Unpool(16 * nf, 8 * nf, 5),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * nf),
            Unpool(8 * nf, 4 * nf, 5),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * nf),
            Unpool(4 * nf, 2 * nf, 5),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * nf),
            Unpool(2 * nf, nf, 5),
            # x1
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
        )

        self.img_mu = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Tanh(),
        )
        self.img_sd = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Softplus(),
        )

        self.xpr_state = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
        )

    @property
    def init_args(self):
        return self._init_args

    def set_factors(self, n):
        w = self.mixture_loadings[-1].weight
        b = self.mixture_loadings[-1].bias

        for p in (w, b):
            while len(p) < n:
                p.data = t.cat([p.data, p[-1:]])
            while len(p) > n:
                p.data = p[:1]

        return self

    def encode(self, img, lbl=None, xpr=None):
        enc_img = self.img_encoder(img)

        if lbl is not None and xpr is not None:
            lbl_16 = (
                t.nn.functional.interpolate(
                    lbl.float().unsqueeze(1),
                    enc_img.shape[-2:],
                )
                .squeeze(1)
                .long()
            )

            missing = t.tensor([1., *[0.] * xpr.shape[1]], device=xpr.device)

            data_with_missing = t.nn.functional.pad(
                xpr, (1, 0, 1, 0))
            data_with_missing[0] = missing

            if self.training:
                data_with_missing[
                    t.distributions.Bernoulli(0.5)
                    .sample((len(data_with_missing), ))
                    .byte()
                ] = missing

            convolved_data = self.xpr_encoder1(data_with_missing)
            xpr = t.einsum(
                'byxi,ic->bcyx',
                (
                    t.eye(len(convolved_data))
                    .to(lbl_16)
                    [lbl_16.flatten()]
                    .reshape(*lbl_16.shape, -1)
                    .float()
                ),
                convolved_data,
            )
        else:
            xpr = t.zeros((
                enc_img.shape[0],
                self.xpr_encoder1.out_features,
                *enc_img.shape[2:],
            )).to(img)
            xpr[:, 0] = 1

        enc_xpr = self.xpr_encoder2(xpr)

        enc_img = center_crop(enc_img, enc_xpr.shape)
        enc_xpr = center_crop(enc_xpr, enc_img.shape)

        x = t.cat([enc_img, enc_xpr], 1)

        z_mu = self.z_mu(x)
        z_sd = self.z_sd(x)

        self.z.distribution.set(
            loc=z_mu,
            scale=z_sd,
            r_transform=True,
        )
        z = self.z.sample().value

        return (
            z,
            z_mu,
            z_sd,
        )

    def decode(self, z):
        state = self.decoder(z)

        img_mu = self.img_mu(state)
        img_sd = self.img_sd(state)

        xpr_state = self.xpr_state(state)

        return (
            t.distributions.Normal(img_mu, img_sd),
            xpr_state,
            state,
        )

    def forward(self, img, lbl, xpr):
        z, z_mu, z_sd = self.encode(img, lbl, xpr)

        def _crop(y):
            if isinstance(y, t.Tensor):
                return center_crop(y, [None, None, *img.shape[-2:]])
            if isinstance(y, t.distributions.Distribution):
                return type(y)(**{
                    k: center_crop(v, [None, None, *img.shape[-2:]])
                    for k, v in y.__dict__.items() if k[0] != '_'
                })
            return y

        return (z, *map(_crop, self.decode(z)))


@store_init_args
class ExpressionProgram(Variational):
    def __init__(self, genes):
        self._make_covariate('rg', (self.factors, len(genes)))
        self._make_covariate(
            'r',
            (self.factors, ),
            pdistr=Beta, pparams={
                'shape1': np.log(np.exp(1) - 1),
                'shape2': np.log(np.exp(1) - 1),
            },
            qdistr=Beta, qparams={
                'shape1': np.log(np.exp(1) - 1),
                'shape2': np.log(np.exp(1) - 1),
            },
        )

    def forward(self, xpr_state):
        pass


@store_init_args
class STD(Variational):
    def __init__(
            self,
            genes: List[str],
            gene_baseline: Optional[np.ndarray] = None,
            covariates: Optional[List[Tuple[str, List[str]]]] = None,
    ):
        super().__init__()

        self.genes = list(genes)

        self._make_covariate('lg', (len(genes), ))
        self._make_covariate('rg', (len(genes), ))

        if covariates is not None and len(covariates) > 0:
            self._covariates = covariates
            n_fe = reduce(op.add, map(lambda x: len(x[1]), covariates))
            self._make_covariate('rgeff', (n_fe, len(genes)))
            self._make_covariate('lgeff', (n_fe, len(genes)))
        else:
            self._covariates = []

        if gene_baseline is not None:
            if len(gene_baseline) != len(genes):
                raise ValueError(
                    'size of `gene_baseline` does not match `genes`'
                    f' ({gene_baseline.shape[1]} vs. {len(genes)})'
                )
            lgb = t.tensor(np.log(gene_baseline)).float()
            self.rg_q_loc.data = lgb.clone()
            self.rg_p_loc.data = lgb.clone()

    @property
    def factor_contrib(self):
        contrib = self.rt.value
        if len(contrib) > 1:
            contrib[1:] *= t.stack([
                *it.accumulate(1 - self.rt.value[:-1], op.mul)])
        return contrib

    def forward(self, x, effects=None):
        rate_tg = (
            (self.rg.value + self.rtg.value).exp()
            * self.factor_contrib[:, None]
        )
        rate = x @ rate_tg
        logit = self.lg.value.unsqueeze(0)
        if effects is not None:
            effects = effects.float()
            rate = rate * (effects @ self.rgeff.value.exp())
            logit = logit + effects @ self.lgeff.value
        return t.distributions.NegativeBinomial(total_count=rate, logits=logit)
