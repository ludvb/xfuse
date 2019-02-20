# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import gzip

import itertools as it

import os

from imageio import imread, imwrite

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch as t

from torchvision.utils import make_grid

DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def zip_dicts(ds):
    d0 = next(ds)
    d = {k: [] for k in d0.keys()}
    for d_ in it.chain([d0], ds):
        for k, v in d_.items():
            try:
                d[k].append(v)
            except AttributeError:
                raise ValueError('dict keys are inconsistent')
    return d


def collect_items(d):
    d_ = {}
    for k, v in d.items():
        try:
            d_[k] = v.item()
        except (ValueError, AttributeError):
            pass
    return d_


def visualize_batch(batch, **kwargs):
    return plt.imshow(
        np.transpose(
            make_grid(
                batch[:64],
                padding=2,
                normalize=True,
            ),
            (1, 2, 0),
        ),
        **kwargs,
    )


def center_crop(input, target_shape):
    return input[tuple([
        slice((a - b) // 2, (a - b) // 2 + b)
        if b is not None else
        slice(None)
        for a, b in zip(input.shape, target_shape)
    ])]


class Histonet(t.nn.Module):
    def __init__(
            self,
            image,
            data,
            hidden_size=512,
            latent_size=96,
            nf=16,
    ):
        super().__init__()

        num_genes = len(data.columns)

        self._shape = [None, None, *image.shape[:2]]

        self.z_mu = t.nn.Parameter(t.zeros(
            [1, latent_size]
            + [np.ceil(x / 16).astype(int) for x in image.shape[:2]]
        ))
        self.z_sd = t.nn.Parameter(t.zeros(
            [1, latent_size]
            + [np.ceil(x / 16).astype(int) for x in image.shape[:2]]
        ))

        self.image_decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 4, 1, 0),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(2 * nf, 3, 4, 2, 1),
            # x1
            t.nn.Tanh(),
        )

        self.transcript_decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 4, 1, 0),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
        )

        self.lrate = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, num_genes, 3, 1, 1),
        )
        self.logit_mu = t.nn.Parameter(
            t.zeros(1, num_genes, 1, 1),
        )
        self.logit_sd = t.nn.Parameter(
            t.zeros(1, num_genes, 1, 1),
        )

    def forward(self):
        z = t.distributions.Normal(self.z_mu, self.z_sd).rsample()

        image = center_crop(self.image_decoder(z), self._shape)
        transcript_state = center_crop(
            self.transcript_decoder(z), self._shape)

        lrate = self.lrate(transcript_state)
        logit = t.distributions.Normal(self.logit_mu, self.logit_sd).rsample()

        return (
            z,
            image,
            lrate,
            logit,
        )


class Discriminator(t.nn.Module):
    def __init__(self, nz, nf=3):
        super().__init__()
        self.downsampler = t.nn.Sequential(
            t.nn.Conv2d(3, nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, 2 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1),
            t.nn.LeakyReLU(0.2, inplace=True),
        )
        self.predictor = t.nn.Sequential(
            t.nn.Conv2d(8 * nf + nz, 16 * nf, 4),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(16 * nf, 16 * nf, 4),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(16 * nf, 1, 4),
        )

    def forward(self, img, z):
        x = self.downsampler(img)
        x = t.cat([x, center_crop(z, [None, None, *x.shape[-2:]])], dim=1)
        x = self.predictor(x)
        return x


def store_state(vae, discriminator, optimizers, iteration, file):
    t.save(
        dict(
            vae=vae.state_dict(),
            discriminator=discriminator.state_dict(),
            optimizers=[x.state_dict() for x in optimizers],
            iteration=iteration,
        ),
        file,
    )


def restore_state(vae, discriminator, optimizers, file):
    state = t.load(file)
    vae.load_state_dict(state['vae'])
    discriminator.load_state_dict(state['discriminator'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    return state['iteration']


def run(
        image: np.array,
        label: np.array,
        data: pd.DataFrame,
        latent_size: int,
        output_prefix,
        state=None,
        report_interval=50,
        spike_prior=False,
        anneal_dkl=False,
):
    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(output_prefix, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    histonet = Histonet(
        image=image,
        data=data,
        latent_size=latent_size,
    ).to(DEVICE)
    discriminator = Discriminator(latent_size).to(DEVICE)

    his_optimizer = t.optim.Adam(
        histonet.parameters(),
        lr=1e-3,
        betas=(0.5, 0.999),
    )
    dis_optimizer = t.optim.Adam(
        discriminator.parameters(),
        lr=2e-3 / 100,
        betas=(0.5, 0.999),
    )

    if state:
        start_epoch = restore_state(
            histonet,
            discriminator,
            [his_optimizer, dis_optimizer],
            state,
        )
    else:
        start_epoch = 1

    obs = (
        t.tensor(data.values)
        .float()
        .t()
        .unsqueeze(0)
        .to(DEVICE)
    )

    image = (
        t.tensor(
            image.transpose(2, 0, 1),
            requires_grad=False,
        )
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )
    image = image / 255 * 2 - 1

    label = t.eye(int(label.max()) + 1)[label.flatten()].to(DEVICE)

    for iteration in it.count(start_epoch):
        dkl_attenuation = (
            t.tanh(t.as_tensor(iteration).float().to(DEVICE) / 1000)
            if anneal_dkl else
            1.0
        )

        def _step():
            z, gen_img, lrate, logit = histonet()

            # -* discriminator loss *-
            limg1 = discriminator(gen_img.detach(), z.detach())
            lreal = discriminator(image, z.detach())

            discriminator_loss = (
                - t.sum(t.nn.functional.logsigmoid(-limg1))
                - t.sum(t.nn.functional.logsigmoid(lreal))
            )

            if t.mean(t.sigmoid(limg1)) >= 0.35:
                dis_optimizer.zero_grad()
                discriminator_loss.backward()
                dis_optimizer.step()
            else:
                print('discriminator is too strong, skipping update')

            # -* generator loss *-
            limg2 = discriminator(gen_img, z.detach())

            rates = (
                (t.exp(lrate).reshape(*lrate.shape[:2], -1) @ label)
                [:, :, 1:]
                + 1e-10
            )
            # ^ NOTE large memory requirements

            # rates = t.stack(
            #     [
            #         t.exp(lrate.masked_select(label == i))
            #         .reshape(len(data.columns), -1)
            #         .sum(1)
            #         for i in data.index
            #     ],
            #     dim=0,
            # )
            # ^ NOTE slow

            # rates = (
            #     t.stack([
            #         t.bincount(label.flatten(), t.exp(lrate[0, i].flatten()))
            #         for i in range(len(data.columns))
            #     ], dim=-1)
            #     [1:]
            # )
            # ^ NOTE gradient for t.bincount not implemented

            d = t.distributions.NegativeBinomial(
                rates,
                logits=logit.reshape(*logit.shape[:2], -1),
            )
            lpobs = d.log_prob(obs)

            dkl = 0.

            generator_loss = (
                - (t.sum(lpobs) + t.sum(t.nn.functional.logsigmoid(limg2)))
                + dkl * dkl_attenuation
            )

            if t.isnan(generator_loss):
                import ipdb; ipdb.set_trace()

            his_optimizer.zero_grad()
            generator_loss.backward()
            his_optimizer.step()

            return collect_items({
                'd-loss':
                discriminator_loss,
                'g-loss':
                generator_loss,
                'p(lab|z)':
                t.mean(t.exp(lpobs)),
                'rmse':
                t.sqrt(t.mean((d.mean - obs) ** 2)),
                'p1(img|z)':
                t.mean(t.sigmoid(limg1)),
                'p2(img|z)':
                t.mean(t.sigmoid(limg2)),
                'dqp':
                dkl,
            })

        t.enable_grad()
        histonet.train()
        discriminator.train()

        def _postprocess(iteration, output):
            print(
                f'iteration {iteration:4d}:',
                '  //  '.join([
                    '{} = {:>9s}'.format(k, f'{v:.2e}')
                    for k, v in output.items()
                ]),
            )

            if iteration % report_interval == 0:
                t.no_grad()
                histonet.eval()

                _, img, *_ = histonet()

                imwrite(
                    os.path.join(img_prefix, f'iteration-{iteration:04d}.jpg'),
                    ((
                        img[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                        + 1
                    ) / 2 * 255).astype(np.uint8),
                )

                # store_state(
                #     histonet,
                #     discriminator,
                #     [his_optimizer, dis_optimizer],
                #     iteration,
                #     os.path.join(
                #         chkpt_prefix,
                #         f'iteration-{iteration:04d}.model',
                #     ),
                # )

                t.enable_grad()
                histonet.train()

        ds = (_postprocess(i, _step()) for i in it.count(1))
        [*ds]
        # _ = {
        #     k: np.mean(vs)
        #     for k, vs in zip_dicts(ds).items()
        # }


def main():
    import argparse as ap

    args = ap.ArgumentParser()

    args.add_argument('data-dir', type=str)

    args.add_argument('--latent-size', type=int, default=100)
    args.add_argument('--spike-prior', action='store_true')
    args.add_argument('--anneal-dkl', action='store_true')

    args.add_argument('--zoom', type=float, default=0.1)

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--report-interval', type=int, default=100)

    opts = vars(args.parse_args())

    data_dir = opts.pop('data-dir')

    image = imread(os.path.join(data_dir, 'image.tif'))
    label = imread(os.path.join(data_dir, 'label.tif'))
    data = pd.read_csv(
        os.path.join(data_dir, 'data.gz'),
        sep=' ',
        header=0,
        index_col=0,
    )

    from scipy.ndimage.interpolation import zoom
    zoom_level = opts.pop('zoom')
    label = zoom(label, (zoom_level, zoom_level), order=0)
    image = zoom(image, (zoom_level, zoom_level, 1))

    data = data[
        data.sum(0)
        [[x for x in data.columns if 'ambiguous' not in x]]
        .sort_values()
        [-50:]
        .index
    ]

    run(image, label, data, **opts)


if __name__ == '__main__':
    main()


if False:
    from imageio import imread
    from scipy.ndimage.interpolation import zoom
    data = pd.read_csv('~/histonet-test-data/data.gz', sep=' ').set_index('n')
    s = data.sum(0)
    data = data[s[[x for x in s.index if 'ambiguous' not in x]].sort_values()[-50:].index]
    lab = imread('~/histonet-test-data/label.tif')
    img = imread('~/histonet-test-data/image.tif')
    lab = zoom(lab, (0.1, 0.1), order=0)
    img = zoom(img, (0.1, 0.1, 1))
