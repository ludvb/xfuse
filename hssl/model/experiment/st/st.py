import itertools as it
from copy import deepcopy
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pyro as p
import torch
import torch.distributions.constraints as constraints
from pyro.contrib.autoname import scope
from pyro.distributions import (  # pylint: disable=no-name-in-module
    Delta,
    NegativeBinomial,
    Normal,
)
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt

from ....logging import DEBUG, INFO, log
from ....session import get
from ....utility import center_crop, find_device, sparseonehot
from ....utility.modules import get_module
from ..image import Image


class FactorDefault(NamedTuple):
    r"""Factor initialization template"""

    scale: float
    profile: Optional[torch.Tensor]


def _encode_factor_name(n: str):
    return f"!!factor!{n}!!"


class ST(Image):
    r"""Spatial Transcriptomics experiment"""

    @property
    def tag(self):
        return "ST"

    def __init__(
        self,
        *args,
        factors: List[FactorDefault] = [],
        default_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__factors: Dict[str, FactorDefault] = {}
        self.__factor_queue: List[str] = []
        for factor in factors:
            self.add_factor(factor)

        self.__default_scale = default_scale

    @property
    def factors(self) -> Dict[str, FactorDefault]:
        r"""Factor initialization templates"""
        return deepcopy(self.__factors)

    def add_factor(self, factor: Optional[FactorDefault] = None):
        r"""
        Adds a new factor, optionally initialized from a
        :class:`FactorDefault`.
        """
        if factor is None:
            factor = FactorDefault(0.0, None)

        if self.__factor_queue != []:
            new_factor = self.__factor_queue.pop()
        else:
            new_factor = f"{len(self.__factors) + 1:d}"
        assert new_factor not in self.__factors

        log(INFO, "adding factor: %s", new_factor)
        self.__factors.setdefault(new_factor, factor)

        return new_factor

    def split_factor(self, factor: str):
        r"""Adds a new factor by splitting an already existing factor."""
        new_factor = self.add_factor(self.factors[factor])

        log(INFO, "copying factor: %s -> %s", factor, new_factor)

        name = _encode_factor_name(factor)
        new_name = _encode_factor_name(new_factor)

        store = get("param_store")

        for pname in [p for p in store.keys() if name in p]:
            new_pname = pname.replace(name, new_name)
            log(DEBUG, "copying param: %s -> %s", pname, new_pname)
            store.setdefault(
                new_pname,
                store[pname].clone().detach(),
                store._constraints[pname],  # pylint: disable=protected-access
            )

        return new_factor

    def remove_factor(self, n, remove_params=False):
        r"""Removes a factor"""
        log(INFO, "removing factor: %s", n)

        try:
            self.__factors.pop(n)
        except KeyError:
            raise ValueError(
                f"attempted to remove factor {n}, which doesn't exist!"
            )

        self.__factor_queue.append(n)

        if remove_params:
            store = get("param_store")
            optim = get("optimizer")
            pname = _encode_factor_name(n)
            for x in [p for p in store.keys() if pname in p]:
                param = store[x].unconstrained()
                del store[x]
                if optim is not None:
                    del optim.optim_objs[param]

    def _get_scale_decoder(self, in_channels):
        def _create_scale_decoder():
            decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, 7, 1, 3),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.Conv2d(in_channels, 1, 1, 1, 1),
                torch.nn.Softplus(),
            )
            torch.nn.init.constant_(decoder[-2].weight, 0.0)
            torch.nn.init.constant_(
                decoder[-2].bias, np.log(np.exp(self.__default_scale) - 1)
            )
            return decoder

        return get_module("scale", _create_scale_decoder)

    def _get_factor_decoder(self, in_channels, n):
        def _create_factor_decoder():
            decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.Conv2d(in_channels, 1, 1, 1, 1),
            )
            torch.nn.init.constant_(decoder[-1].weight, 0.0)
            torch.nn.init.constant_(decoder[-1].bias, self.__factors[n][0])
            return decoder

        return torch.nn.Sequential(
            get_module(
                "factor_predecoder",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(in_channels),
                    torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(in_channels),
                ),
            ),
            get_module(_encode_factor_name(n), _create_factor_decoder),
        )

    def model(self, x, zs):
        # pylint: disable=too-many-locals
        num_genes = x["data"][0].shape[1]

        decoded = self._decode(zs)

        scale = p.sample(
            "scale",
            Delta(
                center_crop(
                    self._get_scale_decoder(decoded.shape[1]).to(decoded)(
                        decoded
                    ),
                    [None, None, *x["label"].shape[-2:]],
                )
            ),
        )

        if len(self.factors) > 0:
            rim = torch.cat(
                [
                    self._get_factor_decoder(decoded.shape[1], n).to(decoded)(
                        decoded
                    )
                    for n in self.factors
                ],
                dim=1,
            )
            rim = center_crop(rim, [None, None, *x["label"].shape[-2:]])
            rim = torch.nn.functional.softmax(rim, dim=1)
            rim = p.sample("rim", Delta(rim))
            rim = scale * rim

            rate_mg_prior = Normal(
                p.param(f"rate_mg_mu", torch.zeros(num_genes)).to(decoded),
                1e-8
                + p.param(
                    f"rate_mg_sd",
                    torch.ones(num_genes),
                    constraint=constraints.positive,
                ).to(decoded),
            )
            rate_mg = torch.stack(
                [
                    p.sample(_encode_factor_name(n), rate_mg_prior)
                    for n in self.factors
                ]
            )
            rate_mg = p.sample("rate_mg", Delta(rate_mg))
        else:
            rim = p.sample(
                "rim",
                Delta(
                    torch.zeros(len(x["data"]), 0, *x["label"].shape[-2:]).to(
                        decoded
                    )
                ),
            )
            rate_mg = p.sample(
                "rate_mg", Delta(torch.zeros(0, num_genes).to(decoded))
            )

        effects = torch.cat(
            [
                torch.ones(x["effects"].shape[0], 1).to(x["effects"]),
                x["effects"],
            ],
            1,
        ).float()
        rate_g_effects = p.sample(
            "rate_g_effects",
            (
                # pylint: disable=not-callable
                Normal(torch.tensor(0.0).to(decoded), 1).expand(
                    [effects.shape[1], num_genes]
                )
            ),
        )
        logits_g_effects = p.sample(
            "logits_g_effects",
            (
                # pylint: disable=not-callable
                Normal(torch.tensor(0.0).to(decoded), 1).expand(
                    [effects.shape[1], num_genes]
                )
            ),
        )

        logits_g = effects @ logits_g_effects
        rate_g = effects @ rate_g_effects
        rate_mg = rate_g[:, None] + rate_mg

        with p.poutine.scale(scale=self.n / len(x["data"])):
            with scope(prefix=self.tag):
                image_distr = self._sample_image(x, decoded)

                def _compute_sample_params(
                    data, label, rim, rate_mg, logits_g
                ):
                    nonmissing = label != 0
                    zero_count_spots = 1 + torch.where(data.sum(1) == 0)[0]
                    nonpartial = binary_fill_holes(
                        np.isin(label.cpu(), [0, *zero_count_spots.cpu()])
                    )
                    nonpartial = torch.as_tensor(nonpartial).to(nonmissing)
                    mask = nonpartial & nonmissing

                    rim = rim[:, mask]
                    label = label[mask] - 1
                    idxs, label = torch.unique(label, return_inverse=True)
                    data = data[idxs]

                    labelonehot = sparseonehot(label)
                    rim = torch.sparse.mm(labelonehot.t().float(), rim.t())
                    rgs = rim @ rate_mg.exp()

                    return data, rgs, logits_g.expand(len(rgs), -1)

                data, rgs, logits_g = zip(
                    *it.starmap(
                        _compute_sample_params,
                        zip(x["data"], x["label"], rim, rate_mg, logits_g),
                    )
                )

                expression_distr = NegativeBinomial(
                    total_count=1e-8 + torch.cat(rgs),
                    logits=torch.cat(logits_g),
                )
                p.sample("xsg", expression_distr, obs=torch.cat(data))

        return image_distr, expression_distr

    def guide(self, x):
        num_genes = x["data"][0].shape[1]

        for name, dim in [
            ("rate_g_effects", [1 + x["effects"].shape[1], num_genes]),
            ("logits_g_effects", [1 + x["effects"].shape[1], num_genes]),
        ]:
            p.sample(
                name,
                Normal(
                    p.param(f"{name}_mu", torch.zeros(dim)).to(find_device(x)),
                    1e-8
                    + p.param(
                        f"{name}_sd",
                        1e-2 * torch.ones(dim),
                        constraint=constraints.positive,
                    ).to(find_device(x)),
                ),
            )

        for n, factor in self.factors.items():
            if factor.profile is None:
                factor = FactorDefault(factor.scale, torch.zeros(num_genes))
            p.sample(
                _encode_factor_name(n),
                Normal(
                    p.param(
                        f"{_encode_factor_name(n)}_mu", factor.profile.float()
                    ).to(find_device(x)),
                    1e-8
                    + p.param(
                        f"{_encode_factor_name(n)}_sd",
                        1e-2 * torch.ones_like(factor.profile).float(),
                        constraint=constraints.positive,
                    ).to(find_device(x)),
                ),
            )

        expression_encoder = get_module(
            "expression_encoder",
            lambda: torch.nn.Sequential(
                torch.nn.Linear(num_genes, 256),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Linear(256, 256),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Linear(256, 16),
            ),
        ).to(x["image"])

        def encode(data, label):
            # Replace missing labels with closest neighbor
            if 0 in label:
                dim1, dim2 = distance_transform_edt(
                    (label == 0).cpu(),
                    return_distances=False,
                    return_indices=True,
                )
                _, n_original = label.unique(return_counts=True)
                label = label[dim1, dim2]
                idxs, n_new = label.unique(return_counts=True)
                scaling = torch.zeros(data.shape[0]).to(data)
                scaling[idxs - 1] = n_original[1:].float() / n_new.float()
                data = data * scaling.unsqueeze(1)
            label = label - 1

            # Expand labels into (encoded) expression vectors
            encdat = expression_encoder(data)
            labelonehot = sparseonehot(label.flatten(), len(encdat))
            expanded = torch.sparse.mm(labelonehot.float(), encdat)
            expanded = expanded.reshape(*label.shape, -1)

            return expanded.permute(2, 0, 1)

        expression = torch.stack(
            [encode(data, label) for data, label in zip(x["data"], x["label"])]
        )
        amended_x = x.copy()
        amended_x["image"] = torch.cat([x["image"], expression], dim=1)

        return super().guide(amended_x)
