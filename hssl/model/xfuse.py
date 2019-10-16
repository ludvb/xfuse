from typing import Dict, List

import pyro as p
from pyro.distributions import Delta, Normal

import torch as t

from .experiment import Experiment
from ..logging import INFO, log
from ..utility import find_device


class XFuse(t.nn.Module):
    def __init__(self, experiments: List[Experiment], latent_size: int):
        super().__init__()
        self.latent_size = latent_size

        self.__experiment_store: Dict[str, Experiment] = {}
        for experiment in experiments:
            self._register_experiment(experiment)

    def _get_experiment(self, experiment_type: str) -> Experiment:
        try:
            return self.__experiment_store[experiment_type]
        except KeyError:
            raise RuntimeError(f"unknown experiment type: {experiment_type}")

    def _register_experiment(self, experiment: Experiment) -> None:
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

    def model(self, xs):
        def _go(e, x):
            with p.poutine.scale(scale=e.n / len(x)):
                z = p.sample(
                    f"z-{e.tag}",
                    (
                        Normal(t.tensor(0.0, device=find_device(x)), 1.0)
                        .expand([1, 1, 1, 1])
                        .to_event(3)
                    ),
                )
            return z, e.model(x, z)

        results = {e: _go(self._get_experiment(e), x) for e, x in xs.items()}
        z = p.sample("z", Delta(t.cat([z for z, _ in results.values()])))
        outputs = {e: output for e, (_, output) in results.items()}
        return z, outputs

    def guide(self, xs):
        def _go(e, x):
            preencoded = e.guide(x)
            z_mu = p.module(
                "z_mu",
                t.nn.Sequential(
                    t.nn.Conv2d(
                        preencoded.shape[1], self.latent_size, 5, 1, 2
                    ),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(self.latent_size),
                    t.nn.Conv2d(self.latent_size, self.latent_size, 5, 1, 2),
                ),
                update_module_params=True,
            ).to(preencoded)
            z_sd = p.module(
                "z_sd",
                t.nn.Sequential(
                    t.nn.Conv2d(
                        preencoded.shape[1], self.latent_size, 5, 1, 2
                    ),
                    t.nn.LeakyReLU(0.2, inplace=True),
                    t.nn.BatchNorm2d(self.latent_size),
                    t.nn.Conv2d(self.latent_size, self.latent_size, 5, 1, 2),
                    t.nn.Softplus(),
                ),
                update_module_params=True,
            ).to(preencoded)
            with p.poutine.scale(scale=e.n / len(x)):
                return p.sample(
                    f"z-{e.tag}",
                    (Normal(z_mu(preencoded), z_sd(preencoded)).to_event(3)),
                )

        p.sample(
            "z",
            Delta(
                t.cat(
                    [_go(self._get_experiment(e), x) for e, x in xs.items()], 0
                )
            ),
        )
