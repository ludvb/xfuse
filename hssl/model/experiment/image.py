import pyro as p
from pyro.distributions import Normal

import torch as t

from . import Experiment
from ...utility import center_crop
from ...utility.misc import Unpool


class Image(Experiment):
    @property
    def tag(self):
        return 'image'

    def _decode(self, z):
        decoder = p.module(
            'img_decoder',
            t.nn.Sequential(
                t.nn.Conv2d(z.shape[1], 512, 5, padding=5),
                # x16
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(512),
                Unpool(512, 256, 5),
                # x8
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(256),
                Unpool(256, 128, 5),
                # x4
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(128),
                Unpool(128, 64, 5),
                # x2
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(64),
                Unpool(64, 32, 5),
                # x1
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(32),
            ),
            update_module_params=True,
        ).to(z)
        return decoder(z)

    def _sample_image(self, x, decoded):
        img_mu = p.module(
            'img_mu',
            t.nn.Sequential(
                t.nn.Conv2d(32, 16, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16),
                t.nn.Conv2d(16, 3, 3, 1, 1),
                t.nn.Tanh(),
            ),
            update_module_params=True,
        ).to(decoded)
        img_sd = p.module(
            'img_sd',
            t.nn.Sequential(
                t.nn.Conv2d(32, 16, 3, 1, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(16),
                t.nn.Conv2d(16, 3, 3, 1, 1),
                t.nn.Softplus(),
            ),
            update_module_params=True,
        ).to(decoded)
        mu = center_crop(img_mu(decoded), [None, None, *x['image'].shape[-2:]])
        sd = center_crop(img_sd(decoded), [None, None, *x['image'].shape[-2:]])

        return p.sample(
            'image',
            Normal(mu, sd).to_event(3),
            obs=x['image'],
        )

    def model(self, x, z):
        decoded = self._decode(z)
        with p.poutine.scale(self.n/len(x)):
            self._sample_image(x, decoded)

    def guide(self, x):
        encoder = p.module(
            'img_encoder',
            t.nn.Sequential(
                # x1
                t.nn.Conv2d(3, 64, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(64),
                # x2
                t.nn.Conv2d(64, 128, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(128),
                # x4
                t.nn.Conv2d(128, 256, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(256),
                # x8
                t.nn.Conv2d(256, 512, 4, 2, 1),
                t.nn.LeakyReLU(0.2, inplace=True),
                t.nn.BatchNorm2d(512),
                # x16
            ),
            update_module_params=True,
        ).to(x['image'])
        return encoder(x['image'])
