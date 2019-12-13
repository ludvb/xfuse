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
                    torch.nn.ConvTranspose2d(
                        y.shape[1], z.shape[1], kernel_size=2, stride=2,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(z.shape[1]),
                ),
            ).to(y)
            decoder2 = get_module(
                f"img-decoder-{i}-2",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(2 * z.shape[1], z.shape[1], kernel_size=3),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(z.shape[1]),
                    torch.nn.Conv2d(z.shape[1], z.shape[1], kernel_size=3),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(z.shape[1]),
                ),
            ).to(y)
            y = decoder1(y)
            z = center_crop(z, [None, None, *y.shape[-2:]])
            return decoder2(torch.cat([y, z], 1))

        y = zs[-1]
        for i, z in enumerate(reversed(zs[:-1])):
            y = _decode(y, z, i)

        return y

    def _encode(self, x):
        def _encode(x, i):
            out_nc = 2 ** i * self.num_channels
            encoder = get_module(
                f"img-encoder-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(x.shape[1], out_nc, kernel_size=3),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(out_nc),
                    torch.nn.Conv2d(out_nc, out_nc, kernel_size=3),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(out_nc),
                ).to(x),
            ).to(x)
            return encoder(x)

        def _downsample(x, i):
            downsampler = get_module(
                f"img-downsampler-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        x.shape[1], 2 * x.shape[1], kernel_size=2, stride=2
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.BatchNorm2d(2 * x.shape[1]),
                ).to(x),
            ).to(x)
            return downsampler(x)

        ys = [_encode(x, 0)]
        for i in range(1, self.depth + 1):
            ys.append(_encode(_downsample(ys[-1], i), i))

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
        mu = img_mu(decoded)
        sd = img_sd(decoded)

        image_distr = Normal(mu, 1e-8 + sd).to_event(3)
        pyro.sample(
            "image",
            image_distr,
            obs=center_crop(x["image"], image_distr.shape()),
        )
        return image_distr

    def model(self, x, zs):
        decoded = self._decode(zs)
        with pyro.poutine.scale(self.n / len(x["image"])):
            return self._sample_image(x, decoded)

    def guide(self, x):
        return self._encode(x["image"])
