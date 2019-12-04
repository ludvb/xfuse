import numpy as np
import pyro
import torch
from pyro.distributions import Normal  # pylint: disable=no-name-in-module

from ...utility import center_crop
from ...utility.modules import get_module
from . import Experiment


class Image(Experiment):
    r"""Image experiment"""

    def __init__(self, *args, depth=4, num_channels=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.num_channels = num_channels

    @property
    def num_z(self):
        return self.depth

    @property
    def tag(self):
        return "image"

    def _decode(self, zs):
        # pylint: disable=no-self-use
        def _decode(y, z, i):
            decoder1 = get_module(
                f"img-decoder-{i}-1",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        y.shape[1],
                        y.shape[1],
                        kernel_size=5,
                        padding=4,
                        dilation=2,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(y.shape[1]),
                    torch.nn.Conv2d(
                        y.shape[1],
                        z.shape[1],
                        kernel_size=5,
                        padding=4,
                        dilation=2,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(z.shape[1]),
                ),
            ).to(y)
            decoder2 = get_module(
                f"img-decoder-{i}-2",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        2 * z.shape[1],
                        2 * z.shape[1],
                        kernel_size=5,
                        padding=2,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(2 * z.shape[1]),
                    torch.nn.Conv2d(
                        2 * z.shape[1], z.shape[1], kernel_size=5, padding=2,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(z.shape[1]),
                ),
            ).to(y)
            y = decoder1(y)
            y = torch.nn.functional.interpolate(y, scale_factor=2.0)
            ypad = max(0, y.shape[-1] - z.shape[-1]) / 2
            xpad = max(0, y.shape[-2] - z.shape[-2]) / 2
            z = torch.nn.functional.pad(
                z,
                (
                    int(np.ceil(ypad)),
                    int(np.floor(ypad)),
                    int(np.ceil(xpad)),
                    int(np.floor(xpad)),
                ),
            )
            y = torch.cat([y, z], 1)
            return decoder2(y)

        y = zs[-1]
        for i, z in enumerate(reversed(zs[:-1])):
            y = _decode(y, z, i)

        return y

    def _encode(self, x):
        def _encode(x, i):
            in_nc = 2 ** i * self.num_channels
            out_nc = 2 ** (i + 1) * self.num_channels
            encoder = get_module(
                f"img-encoder-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(in_nc, out_nc, 4, 2, 1),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(out_nc),
                    torch.nn.Conv2d(out_nc, out_nc, 3, 1, 2),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(out_nc),
                ).to(x),
            ).to(x)
            return encoder(x)

        preencoder = get_module(
            "img-preencoder",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(x.shape[1], self.num_channels, 7, 1, 3),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(self.num_channels),
                torch.nn.Conv2d(self.num_channels, self.num_channels, 7, 1, 3),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(self.num_channels),
            ),
        ).to(x)
        ys = [preencoder(x)]
        for i in range(self.depth):
            ys.append(_encode(ys[-1], i))

        return ys

    def _sample_image(self, x, decoded):
        img_mu = get_module(
            "img_mu",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(self.num_channels, self.num_channels, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(self.num_channels),
                torch.nn.Conv2d(self.num_channels, x["image"].shape[1], 1),
                torch.nn.Tanh(),
            ),
        ).to(decoded)
        img_sd = get_module(
            "img_sd",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(self.num_channels, self.num_channels, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.BatchNorm2d(self.num_channels),
                torch.nn.Conv2d(self.num_channels, x["image"].shape[1], 1),
                torch.nn.Softplus(),
            ),
        ).to(decoded)
        mu = center_crop(img_mu(decoded), [None, None, *x["image"].shape[-2:]])
        sd = center_crop(img_sd(decoded), [None, None, *x["image"].shape[-2:]])

        image_distr = Normal(mu, 1e-8 + sd).to_event(3)
        pyro.sample("image", image_distr, obs=x["image"])
        return image_distr

    def model(self, x, zs):
        decoded = self._decode(zs)
        with pyro.poutine.scale(self.n / len(x["image"])):
            return self._sample_image(x, decoded)

    def guide(self, x):
        return self._encode(x["image"])
