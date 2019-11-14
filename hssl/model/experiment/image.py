from functools import reduce
from operator import add

import pyro as p
import torch as t
from pyro.distributions import Normal  # pylint: disable=no-name-in-module

from ...utility import center_crop
from ...utility.misc import Unpool
from ...utility.modules import get_module
from . import Experiment


class Image(Experiment):
    r"""Image experiment"""

    def __init__(self, *args, depth=4, num_channels=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.num_channels = num_channels

    @property
    def tag(self):
        return "image"

    def _decode(self, zs):
        z = zs[0]
        ncs = [
            2 ** i * self.num_channels for i in reversed(range(self.depth + 1))
        ]
        decoder = get_module(
            "img_decoder",
            lambda: t.nn.Sequential(
                t.nn.Conv2d(z.shape[1], ncs[0], 5, padding=5),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(ncs[0]),
                t.nn.Conv2d(ncs[0], ncs[0], 7, padding=3),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(ncs[0]),
                *reduce(
                    add,
                    [
                        [
                            Unpool(in_nc, out_nc, 5),
                            t.nn.LeakyReLU(0.2, inplace=True),
                            t.nn.BatchNorm2d(out_nc),
                        ]
                        for in_nc, out_nc in zip(ncs, ncs[1:])
                    ],
                ),
            ),
        ).to(z)
        return decoder(z)

    def _sample_image(self, x, decoded):
        img_mu = get_module(
            "img_mu",
            lambda: t.nn.Sequential(
                t.nn.Conv2d(
                    self.num_channels,
                    self.num_channels,
                    x["image"].shape[1],
                    1,
                    1,
                ),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.num_channels),
                t.nn.Conv2d(self.num_channels, x["image"].shape[1], 3, 1, 1),
                t.nn.Tanh(),
            ),
        ).to(decoded)
        img_sd = get_module(
            "img_sd",
            lambda: t.nn.Sequential(
                t.nn.Conv2d(
                    self.num_channels,
                    self.num_channels,
                    x["image"].shape[1],
                    1,
                    1,
                ),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.num_channels),
                t.nn.Conv2d(self.num_channels, x["image"].shape[1], 3, 1, 1),
                t.nn.Softplus(),
            ),
        ).to(decoded)
        mu = center_crop(img_mu(decoded), [None, None, *x["image"].shape[-2:]])
        sd = center_crop(img_sd(decoded), [None, None, *x["image"].shape[-2:]])

        image_distr = Normal(mu, sd).to_event(3)
        p.sample("image", image_distr, obs=x["image"])
        return image_distr

    def model(self, x, zs):
        decoded = self._decode(zs)
        with p.poutine.scale(self.n / len(x["image"])):
            return self._sample_image(x, decoded)

    def guide(self, x):
        ncs = [
            x["image"].shape[1],
            *[2 ** i * self.num_channels for i in range(1, self.depth + 1)],
        ]
        encoder = get_module(
            "img_encoder",
            lambda: t.nn.Sequential(
                *reduce(
                    add,
                    [
                        [
                            t.nn.Conv2d(in_nc, out_nc, 4, 2, 1),
                            t.nn.LeakyReLU(0.2, inplace=True),
                            t.nn.BatchNorm2d(out_nc),
                        ]
                        for in_nc, out_nc in zip(ncs, ncs[1:])
                    ],
                ),
                t.nn.Conv2d(ncs[-1], ncs[-1], 7, 1, 3),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(ncs[-1]),
                t.nn.Conv2d(ncs[-1], ncs[-1], 7, 1, 3),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(ncs[-1]),
            ),
        ).to(x["image"])
        return [encoder(x["image"])]
