import itertools as it
from copy import deepcopy
from functools import partial
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
from ....utility.state import (
    get_module,
    get_param,
    get_state_dict,
    load_state_dict,
)
from ..image import Image


class MetageneDefault(NamedTuple):
    r"""Metagene initialization template"""

    scale: float
    profile: Optional[torch.Tensor]


def _encode_metagene_name(n: str):
    return f"!!metagene!{n}!!"


class ST(Image):
    r"""Spatial Transcriptomics experiment"""

    @property
    def tag(self):
        return "ST"

    def __init__(
        self,
        *args,
        metagenes: Optional[List[MetageneDefault]] = None,
        encode_expression: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if metagenes is None:
            metagenes = [MetageneDefault(0.0, None)]

        if len(metagenes) == 0:
            raise ValueError(f"Needs at least one metagene")

        self.__metagenes: Dict[str, MetageneDefault] = {}
        self.__metagene_queue: List[str] = []
        for metagene in metagenes:
            self.add_metagene(metagene)

        self.__encode_expression = encode_expression

    @property
    def metagenes(self) -> Dict[str, MetageneDefault]:
        r"""Metagene initialization templates"""
        return deepcopy(self.__metagenes)

    def add_metagene(self, metagene: Optional[MetageneDefault] = None):
        r"""
        Adds a new metagene, optionally initialized from a
        :class:`MetageneDefault`.
        """
        if metagene is None:
            metagene = MetageneDefault(0.0, None)

        if self.__metagene_queue != []:
            new_metagene = self.__metagene_queue.pop()
        else:
            new_metagene = f"{len(self.__metagenes) + 1:d}"
        assert new_metagene not in self.__metagenes

        log(INFO, "adding metagene: %s", new_metagene)
        self.__metagenes.setdefault(new_metagene, metagene)

        return new_metagene

    def split_metagene(self, metagene: str):
        r"""Adds a new metagene by splitting an already existing metagene."""
        new_metagene = self.add_metagene(self.metagenes[metagene])

        log(INFO, "copying metagene: %s -> %s", metagene, new_metagene)

        name = _encode_metagene_name(metagene)
        new_name = _encode_metagene_name(new_metagene)

        state_dict = get_state_dict()

        for pname in [
            pname for pname in state_dict.params.keys() if name in pname
        ]:
            new_pname = pname.replace(name, new_name)
            log(DEBUG, "copying param: %s -> %s", pname, new_pname)
            state_dict.params[new_pname] = (
                state_dict.params[pname].detach().clone().requires_grad_()
            )

        for mname in [
            mname for mname in state_dict.modules.keys() if name in mname
        ]:
            new_mname = mname.replace(name, new_name)
            log(DEBUG, "copying module: %s -> %s", mname, new_mname)
            state_dict.modules[new_mname] = state_dict.modules[mname]

        load_state_dict(state_dict)

        return new_metagene

    def remove_metagene(self, n, remove_params=False):
        r"""Removes a metagene"""
        if len(self.metagenes) == 1:
            raise RuntimeError("Cannot remove last metagene")

        log(INFO, "removing metagene: %s", n)

        try:
            self.__metagenes.pop(n)
        except KeyError:
            raise ValueError(
                f"attempted to remove metagene {n}, which doesn't exist!"
            )

        self.__metagene_queue.append(n)

        if remove_params:
            store = p.get_param_store()
            optim = get("optimizer")
            pname = _encode_metagene_name(n)
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
                torch.nn.Conv2d(in_channels, 1, kernel_size=1),
                torch.nn.Softplus(),
            )
            decoder = decoder.to(get("default_device"))
            torch.nn.init.constant_(
                decoder[-2].bias,
                np.log(np.exp(1 / spot_size(dataset)["ST"]) - 1),
            )
            return decoder

        return get_module("scale", _create_scale_decoder)

    def _create_metagene_decoder(self, in_channels, n):
        decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        decoder = decoder.to(get("default_device"))
        torch.nn.init.constant_(decoder[-1].bias, self.__metagenes[n][0])
        return decoder

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

        shared_representation = get_module(
            "metagene_shared",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(
                    decoded.shape[1], decoded.shape[1], kernel_size=1
                ),
                torch.nn.BatchNorm2d(decoded.shape[1]),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ).to(decoded),
        ).to(decoded)(decoded)
        rim = torch.cat(
            [
                get_module(
                    f"decoder_{_encode_metagene_name(n)}",
                    partial(
                        self._create_metagene_decoder, decoded.shape[1], n
                    ),
                ).to(shared_representation)(shared_representation)
                for n in self.metagenes
            ],
            dim=1,
        )
        rim = center_crop(rim, [None, None, *label.shape[-2:]])
        rim = torch.nn.functional.softmax(rim, dim=1)
        rim = p.sample("rim", Delta(rim))
        rim = scale * rim

        rate_mg_prior = Normal(
            get_param(
                f"rate_mg_mu",
                lambda: torch.zeros(num_genes).to(get("default_device")),
            ).to(decoded),
            1e-8
            + get_param(
                f"rate_mg_sd",
                lambda: torch.ones(num_genes).to(get("default_device")),
                constraint=constraints.positive,
            ).to(decoded),
        )
        rate_mg = torch.stack(
            [
                p.sample(_encode_metagene_name(n), rate_mg_prior)
                for n in self.metagenes
            ]
        )
        rate_mg = p.sample("rate_mg", Delta(rate_mg))

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
                            torch.zeros(0, num_genes).to(rim),
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
                torch.nn.Linear(x["data"][0].shape[1], 1024),
                torch.nn.Tanh(),
                torch.nn.Linear(1024, 1024),
                torch.nn.Tanh(),
                torch.nn.Linear(1024, 16),
            ).to(x["image"]),
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
            return (
                torch.as_tensor(data.mean(0))
                .float()
                .squeeze()
                .log()
                .to(device)
            )

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
                        dataset.data.design.shape[0], num_genes, device=device
                    ),
                ).to(device),
                1e-8
                + get_param(
                    f"rate_g_effects-covariate_sd",
                    lambda: 1e-2
                    * torch.ones(
                        dataset.data.design.shape[0], num_genes, device=device
                    ),
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
                    lambda: (0.5 * torch.ones(num_genes, device=device)).log(),
                )
            ),
        ).to(device)
        b = p.sample(
            f"logits_g_effects-covariate",
            Normal(
                get_param(
                    f"logits_g_effects-covariate_mu",
                    lambda: torch.zeros(
                        dataset.data.design.shape[0], num_genes, device=device
                    ),
                ).to(device),
                1e-8
                + get_param(
                    f"logits_g_effects-covariate_sd",
                    lambda: 1e-2
                    * torch.ones(
                        dataset.data.design.shape[0], num_genes, device=device
                    ),
                    constraint=constraints.positive,
                ).to(device),
            ),
        ).to(device)
        p.sample(
            "logits_g_effects",
            Delta(torch.cat([a.unsqueeze(0), b])),
            infer={"is_global": True},
        )

        # Sample metagene profiles
        def _sample_metagene(metagene, name):
            p.sample(
                _encode_metagene_name(name),
                Normal(
                    get_param(
                        f"{_encode_metagene_name(name)}_mu",
                        # pylint: disable=unnecessary-lambda
                        lambda: metagene.profile.float().to(device),
                    ).to(device),
                    1e-8
                    + get_param(
                        f"{_encode_metagene_name(name)}_sd",
                        lambda: 1e-2
                        * torch.ones_like(
                            metagene.profile, device=device
                        ).float(),
                        constraint=constraints.positive,
                    ).to(device),
                ),
                infer={"is_global": True},
            )

        for name, metagene in self.metagenes.items():
            if metagene.profile is None:
                metagene = MetageneDefault(
                    metagene.scale, torch.randn(num_genes)
                )
            _sample_metagene(metagene, name)

    def guide(self, x):
        self._sample_globals()
        if self.__encode_expression:
            expression = self._construct_expression_encoding(x)
            x = x.copy()
            x["image"] = torch.cat([x["image"], expression], dim=1)
        return super().guide(x)
