from typing import Dict, List

import pyro as p
from pyro.distributions import (  # pylint: disable=no-name-in-module
    Delta,
    Normal,
)

import torch

from .experiment import Experiment
from ..logging import INFO, log
from ..utility import find_device
from ..utility.modules import get_module


class XFuse(torch.nn.Module):
    r"""XFuse"""

    def __init__(self, experiments: List[Experiment], latent_size: int):
        super().__init__()
        self.latent_size = latent_size

        self.__experiment_store: Dict[str, Experiment] = {}
        for experiment in experiments:
            self.register_experiment(experiment)

    def get_experiment(self, experiment_type: str) -> Experiment:
        r"""Get registered :class:`Experiment` by tag"""

        try:
            return self.__experiment_store[experiment_type]
        except KeyError:
            raise RuntimeError(f"unknown experiment type: {experiment_type}")

    def register_experiment(self, experiment: Experiment) -> None:
        r"""Registered :class:`Experiment`"""

        if experiment.tag in self.__experiment_store:
            raise RuntimeError(
                f'model for data type "{experiment.tag}" already registered'
            )
        log(
            INFO,
            'registering experiment: %s (data type: "%s")',
            type(experiment).__name__,
            experiment.tag,
        )
        self.__experiment_store[experiment.tag] = experiment

    def forward(self, *input):
        # pylint: disable=redefined-builtin
        return self.model(*input)

    def model(self, xs):
        r"""Runs XFuse on the given data"""

        def _go(experiment, x):
            with p.poutine.scale(scale=experiment.n / len(x)):
                z = p.sample(
                    f"z-{experiment.tag}",
                    (
                        # pylint: disable=not-callable
                        Normal(torch.tensor(0.0, device=find_device(x)), 1.0)
                        .expand([1, self.latent_size, 1, 1])
                        .to_event(3)
                    ),
                )
            return z, experiment.model(x, z)

        results = {e: _go(self.get_experiment(e), x) for e, x in xs.items()}
        z = p.sample("z", Delta(torch.cat([z for z, _ in results.values()])))
        outputs = {e: output for e, (_, output) in results.items()}
        return z, outputs

    def guide(self, xs):
        r"""
        Runs the :class:`pyro.infer.SVI` `guide` for XFuse on the given data
        """

        def _go(experiment, x):
            preencoded = experiment.guide(x)
            z_mu = get_module(
                "z_mu",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        preencoded.shape[1], self.latent_size, 5, 1, 2
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(self.latent_size),
                    torch.nn.Conv2d(
                        self.latent_size, self.latent_size, 5, 1, 2
                    ),
                ),
            ).to(preencoded)
            z_sd = get_module(
                "z_sd",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        preencoded.shape[1], self.latent_size, 5, 1, 2
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(self.latent_size),
                    torch.nn.Conv2d(
                        self.latent_size, self.latent_size, 5, 1, 2
                    ),
                    torch.nn.Softplus(),
                ),
            ).to(preencoded)
            with p.poutine.scale(scale=experiment.n / len(x)):
                return p.sample(
                    f"z-{experiment.tag}",
                    (Normal(z_mu(preencoded), z_sd(preencoded)).to_event(3)),
                )

        p.sample(
            "z",
            Delta(
                torch.cat(
                    [_go(self.get_experiment(e), x) for e, x in xs.items()], 0
                )
            ),
        )
