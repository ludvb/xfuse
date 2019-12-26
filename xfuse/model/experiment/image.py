import pyro
import torch
from actnorm import ActNorm2d
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
        def _decode(y, i):
            decoder = get_module(
                f"img-decoder-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        y.shape[1], y.shape[1], kernel_size=3, padding=1
                    ),
                    ActNorm2d(y.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.Conv2d(
                        y.shape[1], y.shape[1], kernel_size=3, padding=1
                    ),
                    ActNorm2d(y.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ),
            ).to(y)
            return decoder(y)

        def _combine(y, z, i):
            combiner = get_module(
                f"img-combiner-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        y.shape[1] + z.shape[1],
                        z.shape[1],
                        kernel_size=3,
                        padding=1,
                    ),
                    ActNorm2d(z.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ),
            ).to(y)
            y = center_crop(y, [None, None, *z.shape[-2:]])
            return combiner(torch.cat([y, z], 1))

        def _upsample(y, i):
            upsampler = get_module(
                f"upsampler-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Upsample(
                        scale_factor=2.0, mode="bilinear", align_corners=False
                    ),
                    torch.nn.Conv2d(
                        y.shape[1], y.shape[1] // 2, kernel_size=5, padding=2
                    ),
                    ActNorm2d(y.shape[1] // 2),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ),
            ).to(y)
            return upsampler(y)

        y = _decode(zs[-1], self.depth - 1)
        for i, z in zip(reversed(range(self.depth - 1)), zs[::-1][1:]):
            y = _decode(_combine(_upsample(y, i), z, i), i)

        return y

    def _encode(self, x):
        def _encode(x, i):
            encoder = get_module(
                f"encoder-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        x.shape[1], x.shape[1], kernel_size=3, padding=1
                    ),
                    ActNorm2d(x.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.Conv2d(
                        x.shape[1], x.shape[1], kernel_size=3, padding=1
                    ),
                    ActNorm2d(x.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ).to(x),
            ).to(x)
            return encoder(x)

        def _downsample(x, i):
            downsampler = get_module(
                f"downsampler-{i}",
                lambda: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        x.shape[1],
                        2 * x.shape[1],
                        kernel_size=4,
                        stride=2,
                        padding=2,
                    ),
                    ActNorm2d(2 * x.shape[1]),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ).to(x),
            ).to(x)
            return downsampler(x)

        preencoder = get_module(
            "preencoder",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(
                    x.shape[1], self.num_channels, kernel_size=3, padding=1
                ),
                ActNorm2d(self.num_channels),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ).to(x),
        ).to(x)

        ys = [_encode(preencoder(x), 0)]
        for i in range(1, self.depth):
            ys.append(_encode(_downsample(ys[-1], i), i))

        return ys

    def _sample_image(self, x, decoded):
        img_mu = get_module(
            "img_mu",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.num_channels, self.num_channels, kernel_size=1
                ),
                ActNorm2d(self.num_channels),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(
                    self.num_channels, x["image"].shape[1], kernel_size=1
                ),
                torch.nn.Tanh(),
            ),
        ).to(decoded)
        img_sd = get_module(
            "img_sd",
            lambda: torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.num_channels, self.num_channels, kernel_size=1
                ),
                ActNorm2d(self.num_channels),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(
                    self.num_channels, x["image"].shape[1], kernel_size=1
                ),
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
