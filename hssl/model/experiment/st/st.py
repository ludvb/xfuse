from copy import deepcopy

import itertools as it

from typing import Dict, List, NamedTuple, Optional

import numpy as np

import pyro as p
from pyro.distributions import Delta, NegativeBinomial, Normal
from pyro.contrib.autoname import scope

from scipy.ndimage.morphology import distance_transform_edt

import torch as t

from ..image import Image
from ....logging import DEBUG, INFO, log
from ....utility import center_crop, find_device, sparseonehot
from ....session import get_param_store


class FactorDefault(NamedTuple):
    scale: float
    profile: Optional[t.Tensor]


def _encode_factor_name(n: str):
    return f'!!factor!{n}!!'


class ST(Image):
    @property
    def tag(self):
        return 'ST'

    def __init__(
            self,
            *args,
            factors: List[FactorDefault] = [],
            default_scale: float = 1.,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__factors: Dict[str, FactorDefault] = {}
        self.__factors_counter = map(str, it.count())
        for factor in factors:
            self.add_factor(factor)

        self.__default_scale = default_scale

    @property
    def factors(self) -> Dict[str, FactorDefault]:
        return deepcopy(self.__factors)

    def add_factor(self, factor: Optional[FactorDefault] = None):
        if factor is None:
            factor = FactorDefault(0., None)

        new_factor = next(self.__factors_counter)
        assert new_factor not in self.__factors

        log(INFO, 'adding factor: %s', new_factor)
        self.__factors.setdefault(new_factor, factor)

        return new_factor

    def split_factor(self, factor: str):
        new_factor = self.add_factor(self.factors[factor])
        log(INFO, 'splitting factor: %s -> %s', factor, new_factor)

        name = _encode_factor_name(factor)
        new_name = _encode_factor_name(new_factor)

        store = get_param_store()

        for pname in [p for p in store.keys() if name in p]:
            new_pname = pname.replace(name, new_name)
            log(DEBUG, 'copying param: %s -> %s', pname, new_pname)
            store.setdefault(
                new_pname,
                store[pname].clone().detach(),
                store._constraints[pname],
            )

        for b in (self._get_factor_decoder(1, x)[-1].bias
                  for x in (factor, new_factor)):
            b.data -= np.log(2)

        return new_factor

    def remove_factor(self, n, remove_params=False):
        log(INFO, 'removing factor: %s', n)

        try:
            self.__factors.pop(n)
        except KeyError:
            raise ValueError(
                f'attempted to remove factor {n}, which doesn\'t exist!')

        self.__factors_counter = it.chain([n], self.__factors_counter)

        if remove_params:
            store = get_param_store()
            pname = _encode_factor_name(n)
            for param in [p for p in store.keys() if pname in p]:
                del store[param]

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
        return p.module(
            _encode_factor_name(n),
            decoder,
            update_module_params=True,
        )

    def model(self, x, z):
        num_genes = x['data'][0].shape[1]

        decoded = self._decode(z)

        scale = p.sample('scale', Delta(
            center_crop(
                self._get_scale_decoder(decoded.shape[1]).to(decoded)(decoded),
                [None, None, *x['label'].shape[-2:]],
            )
        ))

        if len(self.factors) > 0:
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
            rim = p.sample('rim', Delta(rim))
            rim = scale * rim

            rmg = p.sample('rmg', Delta(t.stack([
                p.sample(_encode_factor_name(n), (
                    Normal(t.tensor(0.).to(z), 1.).expand([num_genes])
                ))
                for n in self.factors
            ])))
        else:
            rim = p.sample('rim', Delta(
                t.zeros(len(x['data']), 0, *x['label'].shape[-2:])
                .to(decoded)
            ))
            rmg = p.sample('rmg', Delta(t.zeros(0, num_genes).to(decoded)))

        effects = x['effects'].float()
        rgeff = p.sample('rgeff', (
            Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))
        lgeff = p.sample('lgeff', (
            Normal(t.tensor(0.).to(z), 1)
            .expand([effects.shape[1], num_genes])
        ))

        lg = effects @ lgeff
        rg = effects @ rgeff
        rmg = rg[:, None] + rmg

        with p.poutine.scale(scale=self.n/len(x['data'])):
            with scope(prefix=self.tag):
                image_distr = self._sample_image(x, decoded)

                def _compute_sample_params(label, rim, rmg, lg):
                    labelonehot = sparseonehot(label.flatten())
                    rim = t.sparse.mm(
                        labelonehot.t().float(),
                        rim.permute(1, 2, 0).view(
                            rim.shape[1] * rim.shape[2],
                            rim.shape[0],
                        ),
                    )
                    rgs = rim[1:] @ rmg.exp()
                    return rgs, lg.expand(len(rgs), -1)

                rgs, lg = zip(*it.starmap(
                    _compute_sample_params, zip(x['label'], rim, rmg, lg)))

                expression_distr = NegativeBinomial(
                    total_count=t.cat(rgs), logits=t.cat(lg))
                p.sample('xsg', expression_distr, obs=t.cat(x['data']))

        return image_distr, expression_distr

    def guide(self, x):
        num_genes = x['data'][0].shape[1]

        for name, dim in [
            ('rgeff', [x['effects'].shape[1], num_genes]),
            ('lgeff', [x['effects'].shape[1], num_genes]),
        ]:
            p.sample(
                name,
                Normal(
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

        for n, factor in self.factors.items():
            if factor.profile is None:
                factor = FactorDefault(factor.scale, t.zeros(num_genes))
            p.sample(
                _encode_factor_name(n),
                Normal(
                    p.param(
                        f'{_encode_factor_name(n)}_mu',
                        factor.profile.float(),
                    ).to(find_device(x)),
                    p.param(
                        f'{_encode_factor_name(n)}_sd',
                        1e-2 * t.ones_like(factor.profile).float(),
                        constraint=t.distributions.constraints.positive,
                    ).to(find_device(x)),
                ),
            )

        image = super().guide(x)

        expression_encoder = p.module(
            'expression_encoder',
            t.nn.Sequential(
                t.nn.Linear(num_genes, 100),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm1d(100),
                t.nn.Linear(100, 100),
            ),
            update_module_params=True,
        ).to(image)

        def encode(data, label):
            encdat = expression_encoder(data)
            missing = t.tensor([1., *[0.] * encdat.shape[1]]).to(encdat)
            encdat_with_missing = t.nn.functional.pad(encdat, (1, 0, 1, 0))
            encdat_with_missing[0] = missing
            labelonehot = sparseonehot(
                label.flatten(), len(encdat_with_missing))
            expanded = t.sparse.mm(labelonehot.float(), encdat_with_missing)
            expanded = expanded.reshape(*label.shape, -1)
            d1, d2 = distance_transform_edt(
                expanded[..., 0].detach().cpu(),
                return_distances=False,
                return_indices=True,
            )
            expanded[..., 1:] = expanded[d1, d2, 1:]
            return expanded.permute(2, 0, 1)

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
