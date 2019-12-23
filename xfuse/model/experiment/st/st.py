import itertools as it
import warnings
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
from scipy.sparse import vstack

from ....data.utility.misc import spot_size
from ....logging import DEBUG, INFO, log
from ....session import get, require
from ....utility import center_crop, sparseonehot
from ....utility.modules import get_module, get_param
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
        encode_expression: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__factors: Dict[str, FactorDefault] = {}
        self.__factor_queue: List[str] = []
        for factor in factors:
            self.add_factor(factor)

        self.__encode_expression = encode_expression

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

        store = p.get_param_store()

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
            store = p.get_param_store()
            optim = get("optimizer")
            pname = _encode_factor_name(n)
            for x in [p for p in store.keys() if pname in p]:
                param = store[x].unconstrained()
                del store[x]
                if optim is not None:
                    del optim.optim_objs[param]

    def _get_scale_decoder(self, in_channels):
        # pylint: disable=no-self-use
        def _create_scale_decoder():
            dataset = require("dataloader").dataset
            decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
                torch.nn.Softplus(),
            )
            torch.nn.init.constant_(decoder[-2].weight, 0.0)
            torch.nn.init.constant_(
                decoder[-2].bias,
                np.log(np.exp(1 / spot_size(dataset)["ST"]) - 1),
            )
            return decoder

        return get_module("scale", _create_scale_decoder)

    def _get_factor_decoder(self, in_channels, n):
        def _create_factor_decoder():
            decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 1, kernel_size=1)
            )
            torch.nn.init.constant_(decoder[-1].weight, 0.0)
            torch.nn.init.constant_(decoder[-1].bias, self.__factors[n][0])
            return decoder

        return torch.nn.Sequential(
            get_module(
                "factor_shared",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels, in_channels, kernel_size=3, padding=1
                    ),
                    torch.nn.BatchNorm2d(in_channels),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ),
            ),
            get_module(_encode_factor_name(n), _create_factor_decoder),
        )

    def model(self, x, zs):
        # pylint: disable=too-many-locals
        num_genes = x["data"][0].shape[1]

        decoded = self._decode(zs)
        label = center_crop(x["label"], [None, *decoded.shape[-2:]])

        scale = p.sample(
            "scale",
            Delta(
                center_crop(
                    self._get_scale_decoder(decoded.shape[1]).to(decoded)(
                        decoded
                    ),
                    [None, None, *label.shape[-2:]],
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
            rim = center_crop(rim, [None, None, *label.shape[-2:]])
            rim = torch.nn.functional.softmax(rim, dim=1)
            rim = p.sample("rim", Delta(rim))
            rim = scale * rim

            rate_mg_prior = Normal(
                get_param(f"rate_mg_mu", lambda: torch.zeros(num_genes)).to(
                    decoded
                ),
                1e-8
                + get_param(
                    f"rate_mg_sd",
                    lambda: torch.ones(num_genes),
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
                    torch.zeros(len(x["data"]), 0, *label.shape[-2:]).to(
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

                    if not mask.any():
                        return (
                            data[[]],
                            torch.zeros(0, num_genes),
                            logits_g.expand(0, -1),
                        )

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
                        zip(x["data"], label, rim, rate_mg, logits_g),
                    )
                )

                expression_distr = NegativeBinomial(
                    total_count=1e-8 + torch.cat(rgs),
                    logits=torch.cat(logits_g),
                )
                p.sample("xsg", expression_distr, obs=torch.cat(data))

        return image_distr, expression_distr

    @staticmethod
    def _construct_expression_encoding(x):
        expression_encoder1 = get_module(
            "expression_encoder1",
            lambda: torch.nn.Sequential(
                torch.nn.Linear(x["data"][0].shape[1], 256),
                torch.nn.Tanh(),
                torch.nn.Linear(256, 256),
                torch.nn.Tanh(),
                torch.nn.Linear(256, 8),
            ),
        ).to(x["image"])

        smoothing_kernel = torch.nn.Conv2d(8, 8, 9, 1, 4, bias=False)
        smoothing_kernel.weight = torch.nn.Parameter(
            torch.ones_like(smoothing_kernel.weight) / 8 / 9 / 9,
            requires_grad=False,
        )
        with warnings.catch_warnings():
            # Ignore Pyro warning that the weights of the `smoothing_kernel`
            # will not be registered to the param store, which is intended.
            warnings.simplefilter("ignore", category=UserWarning)
            expression_encoder2 = get_module(
                "expression_encoder2",
                lambda: torch.nn.Sequential(
                    smoothing_kernel,
                    torch.nn.Conv2d(
                        8, 8, kernel_size=7, stride=1, padding=6, dilation=2
                    ),
                    torch.nn.Tanh(),
                    smoothing_kernel,
                    torch.nn.Conv2d(
                        8, 8, kernel_size=7, stride=1, padding=6, dilation=2
                    ),
                    torch.nn.Tanh(),
                    smoothing_kernel,
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
            encdat = expression_encoder1(data)
            labelonehot = sparseonehot(label.flatten(), len(encdat))
            expanded = torch.sparse.mm(labelonehot.float(), encdat)
            expanded = expanded.reshape(*label.shape, -1)

            return expanded.permute(2, 0, 1)

        def _normalize(data, means, stdvs):
            return (data - means) / stdvs.clamp_min(1e-2)

        expression = torch.stack(
            [
                encode(_normalize(data, means, stdvs), label)
                for data, means, stdvs, label in zip(
                    x["data"], x["means"], x["stdvs"], x["label"]
                )
            ]
        )
        expression = expression_encoder2(expression)

        return expression

    def _sample_globals(self):
        dataset = require("dataloader").dataset
        device = get("default_device")
        num_genes = len(dataset.genes)

        # Sample rate coefficients
        def _init_gene_baselines():
            data = vstack(
                [slide.data.counts for slide in dataset.data.slides.values()]
            )
            return torch.as_tensor(data.mean(0)).float().squeeze().log()

        a = p.sample(
            f"rate_g_effects-baseline",
            Delta(
                get_param(
                    f"rate_g_effects-baseline-value", _init_gene_baselines
                )
            ),
        ).to(device)
        b = p.sample(
            f"rate_g_effects-covariate",
            Normal(
                get_param(
                    f"rate_g_effects-covariate_mu",
                    lambda: torch.zeros(
                        dataset.data.design.shape[0], num_genes
                    ),
                ).to(device),
                1e-8
                + get_param(
                    f"rate_g_effects-covariate_sd",
                    lambda: 1e-2
                    * torch.ones(dataset.data.design.shape[0], num_genes),
                    constraint=constraints.positive,
                ).to(device),
            ),
        ).to(device)
        p.sample(
            "rate_g_effects",
            Delta(torch.cat([a.unsqueeze(0), b])),
            infer={"is_global": True},
        )

        # Sample logits coefficients
        a = p.sample(
            f"logits_g_effects-baseline",
            Delta(
                get_param(
                    f"logits_g_effects-baseline-value",
                    # pylint: disable=unnecessary-lambda
                    lambda: (0.5 * torch.ones(num_genes)).log(),
                )
            ),
        ).to(device)
        b = p.sample(
            f"logits_g_effects-covariate",
            Normal(
                get_param(
                    f"logits_g_effects-covariate_mu",
                    lambda: torch.zeros(
                        dataset.data.design.shape[0], num_genes
                    ),
                ).to(device),
                1e-8
                + get_param(
                    f"logits_g_effects-covariate_sd",
                    lambda: 1e-2
                    * torch.ones(dataset.data.design.shape[0], num_genes),
                    constraint=constraints.positive,
                ).to(device),
            ),
        ).to(device)
        p.sample(
            "logits_g_effects",
            Delta(torch.cat([a.unsqueeze(0), b])),
            infer={"is_global": True},
        )

        # Sample factor profiles
        def _sample_factor(factor, name):
            p.sample(
                _encode_factor_name(name),
                Normal(
                    get_param(
                        f"{_encode_factor_name(name)}_mu",
                        # pylint: disable=unnecessary-lambda
                        lambda: factor.profile.float(),
                    ).to(device),
                    1e-8
                    + get_param(
                        f"{_encode_factor_name(name)}_sd",
                        lambda: 1e-2 * torch.ones_like(factor.profile).float(),
                        constraint=constraints.positive,
                    ).to(device),
                ),
                infer={"is_global": True},
            )

        for name, factor in self.factors.items():
            if factor.profile is None:
                factor = FactorDefault(factor.scale, torch.zeros(num_genes))
            _sample_factor(factor, name)

    def guide(self, x):
        self._sample_globals()
        if self.__encode_expression:
            expression = self._construct_expression_encoding(x)
            x = x.copy()
            x["image"] = torch.cat([x["image"], expression], dim=1)
        return super().guide(x)
