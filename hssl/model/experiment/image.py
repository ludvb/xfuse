import pyro as p
from pyro.distributions import Normal

import torch as t

from . import Experiment
from ...utility import center_crop
from ...utility.misc import Unpool


class Image(Experiment):
    def __init__(self, *args, nc=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = nc

    @property
    def tag(self):
        return 'image'

    def _decode(self, z):
        decoder = p.module(
            'img_decoder',
            t.nn.Sequential(
                t.nn.Conv2d(z.shape[1], 16 * self.nc, 5, padding=5),
                # x16
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16 * self.nc),
                Unpool(16 * self.nc, 8 * self.nc, 5),
                # x8
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(8 * self.nc),
                Unpool(8 * self.nc, 4 * self.nc, 5),
                # x4
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(4 * self.nc),
                Unpool(4 * self.nc, 2 * self.nc, 5),
                # x2
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(2 * self.nc),
                Unpool(2 * self.nc, self.nc, 5),
                # x1
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.nc),
            ),
            update_module_params=True,
        ).to(z)
        return decoder(z)

    def _sample_image(self, x, decoded):
        img_mu = p.module(
            'img_mu',
            t.nn.Sequential(
                t.nn.Conv2d(self.nc, self.nc, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.nc),
                t.nn.Conv2d(self.nc, 3, 3, 1, 1),
                t.nn.Tanh(),
            ),
            update_module_params=True,
        ).to(decoded)
        img_sd = p.module(
            'img_sd',
            t.nn.Sequential(
                t.nn.Conv2d(self.nc, self.nc, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.nc),
                t.nn.Conv2d(self.nc, 3, 3, 1, 1),
                t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)
        mu = center_crop(img_mu(decoded), [None, None, *x['image'].shape[-2:]])
        sd = center_crop(img_sd(decoded), [None, None, *x['image'].shape[-2:]])

        image_distr = Normal(mu, sd).to_event(3)
        p.sample('image', image_distr, obs=x['image'])
        return image_distr

    def model(self, x, z):
        decoded = self._decode(z)
        with p.poutine.scale(self.n/len(x['image'])):
            return self._sample_image(x, decoded)

    def guide(self, x):
        encoder = p.module(
            'img_encoder',
            t.nn.Sequential(
                # x1
                t.nn.Conv2d(3, self.nc, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(self.nc),
                # x2
                t.nn.Conv2d(self.nc, 2 * self.nc, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(2 * self.nc),
                # x4
                t.nn.Conv2d(2 * self.nc, 4 * self.nc, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(4 * self.nc),
                # x8
                t.nn.Conv2d(4 * self.nc, 8 * self.nc, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(8 * self.nc),
                # x16
            ),
            update_module_params=True,
        ).to(x['image'])
        return encoder(x['image'])
