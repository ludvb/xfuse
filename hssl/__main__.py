# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import gzip

import itertools as it

import os

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


class Histonet(t.nn.Module):
    def __init__(
            self,
            num_genes,
            hidden_size=512,
            latent_size=96,
            nf=64,
    ):
        super().__init__()

        self.encoder = t.nn.Sequential(
            t.nn.Conv2d(3 + num_genes, 2 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(2 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(4 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(8 * nf, 16 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(16 * nf, latent_size, 4, 1, 0),
        )

        self.z_mu = t.nn.Conv2d(hidden_size, latent_size, 1)
        self.z_sd = t.nn.Conv2d(hidden_size, latent_size, 1)

        self.image_decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 4, 1, 0),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(4 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(2 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(2 * nf, 3, 4, 2, 1),
            t.nn.Tanh(),
        )

        self.transcriptome_decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 4, 1, 0),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(4 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(2 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1),
            t.nn.BatchNorm2d(num_genes),
            t.nn.LeakyReLU(0.2),
        )

        self.lrate = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1),
            t.nn.BatchNorm2d(num_genes),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(nf, num_genes, 3, 1, 1),
        )
        self.logit = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1),
            t.nn.BatchNorm2d(num_genes),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(nf, num_genes, 3, 1, 1),
        )

    def encode(self, x, label):
        x = x.reshape(x.shape[0], -1)
        x = t.cat([x, t.eye(11)[label].to(DEVICE)], 1)

        x = self.encoder(x)

        z_mu = self.z_mu(x)
        z_sd = t.nn.functional.softplus(self.z_sd(x))

        return z_mu, z_sd

    def decode(self, z):
        img = self.image_decoder(z)

        transcriptome_state = self.transcriptome_decoder(z)

        lrate = self.lrate(transcriptome_state)
        logit = self.logit(transcriptome_state)

        return img, t.distributions.NegativeBinomial(
            t.exp(lrate) + 1e-10, logits=logit)

    def forward(self, x, label):
        z_mu, z_sd = self.encode(x, label)
        z = t.distributions.Normal(z_mu, z_sd).rsample()
        y1, y2 = self.decode(z)
        return z, z_mu, z_sd, y1, y2


class Discriminator(t.nn.Module):
    def __init__(self, input_size, nf=3):
        super().__init__()
        self.discriminator = t.nn.Sequential(
            t.nn.Conv2d(3, nf, 4, 2, 1),
            t.nn.BatchNorm2d(nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(nf, 2 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(2 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(4 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(8 * nf, 1, 4),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


def store_state(vae, discriminator, optimizers, epoch, file):
    t.save(
        dict(
            vae=vae.state_dict(),
            discriminator=discriminator.state_dict(),
            optimizers=[x.state_dict() for x in optimizers],
            epoch=epoch,
        ),
        file,
    )


def restore_state(vae, discriminator, optimizers, file):
    state = t.load(file)
    vae.load_state_dict(state['vae'])
    discriminator.load_state_dict(state['discriminator'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    return state['epoch']


def run(
        image, label, data,
        latent_size,
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

    vae = t.nn.DataParallel(Histonet(
        num_genes=len(data.columns),
        latent_size=latent_size,
    ).to(DEVICE))
    discriminator = t.nn.DataParallel(
        Discriminator(28 * 28 + latent_size).to(DEVICE))

    vae_optimizer = t.optim.Adam(
        vae.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
        # weight_decay=1e-8,
    )
    dis_optimizer = t.optim.Adam(
        discriminator.parameters(),
        lr=2e-4 / 10,
        betas=(0.5, 0.999),
        # weight_decay=1e-8,
    )

    if state:
        start_epoch = restore_state(
            vae,
            discriminator,
            [vae_optimizer, dis_optimizer],
            state,
        )
    else:
        start_epoch = 1

    fixed_noise = t.randn((64, latent_size)).to(DEVICE)

    for epoch in it.count(start_epoch):
        dkl_attenuation = (
            t.tanh(t.as_tensor(epoch).float().to(DEVICE) / 1000)
            if anneal_dkl else
            1.0
        )

        def _step(x, label, observed_label):
            x = x.to(DEVICE)
            label = label.to(DEVICE)
            observed_label = observed_label.to(DEVICE)

            z, z_mu, z_sd, y1, y2 = vae(x, observed_label)

            yz = t.cat([y1.reshape(y1.shape[0], -1), z], 1)

            # -* discriminator loss *-
            limg1 = discriminator(yz.detach())
            lreal = discriminator(
                t.cat([x.reshape(x.shape[0], -1), z.detach()], 1))

            discriminator_loss = (
                - t.sum(t.nn.functional.logsigmoid(-limg1))
                - t.sum(t.nn.functional.logsigmoid(lreal))
            )

            if t.mean(t.sigmoid(limg1)) >= 0.45:
                dis_optimizer.zero_grad()
                discriminator_loss.backward()
                dis_optimizer.step()
            else:
                print('discriminator is too strong, skipping update')

            # -* generator loss *-
            limg2 = discriminator(yz)
            plabl = y2[range(len(y2)), label].masked_select(
                observed_label != 10)

            dkl = t.sum(
                t.distributions.Normal(z_mu, z_sd).log_prob(z)
                - (
                    np.log(0.5)
                    +
                    t.logsumexp(
                        t.stack((
                            t.distributions.Normal(0., 10.).log_prob(z),
                            t.distributions.Normal(0., .1).log_prob(z),
                        )),
                        dim=0,
                    )
                    if spike_prior else
                    t.distributions.Normal(0., 1.).log_prob(z)
                )
            )

            generator_loss = (
                - (t.sum(plabl) + t.sum(t.nn.functional.logsigmoid(limg2)))
                + dkl * dkl_attenuation
            )

            vae_optimizer.zero_grad()
            generator_loss.backward()
            vae_optimizer.step()

            correct = (t.argmax(y2, 1)) == label

            return (
                len(x),
                collect_items({
                    'd-loss':
                    discriminator_loss,
                    'g-loss':
                    generator_loss,
                    'p(lab|z)':
                    t.mean(t.exp(plabl)),
                    'p1(img|z)':
                    t.mean(t.sigmoid(limg1)),
                    'p2(img|z)':
                    t.mean(t.sigmoid(limg2)),
                    'dqp':
                    dkl,
                    'obs. acc':
                    t.mean(correct.masked_select(
                        observed_label != 10).float()),
                    # 'unobs. acc':
                    # t.mean(correct.masked_select(
                    #     observed_label == 10).float()),
                }),
            )

        t.enable_grad()
        vae.train()
        discriminator.train()

        bs, ds = zip(*it.starmap(_step, dataloader))
        output = {
            k: np.sum(np.array(bs) * vs) / len(dataset)
            for k, vs in zip_dicts(iter(ds)).items()
        }
        print(
            f'epoch {epoch:4d}:',
            '  //  '.join([
                '{} = {:>9s}'.format(k, f'{v:.2e}')
                for k, v in output.items()
            ]),
        )

        with gzip.open(
                os.path.join(output_prefix, 'training_data.csv.gz'),
                'ab',
        ) as fh:
            fh.writelines([
                f'{epoch},{k},{v}\n'.encode()
                for k, v in output.items()
            ])

        if epoch % report_interval == 0:
            t.no_grad()
            vae.eval()

            imgs, logits = vae.module.decode(fixed_noise)

            visualize_batch(imgs.detach().cpu())
            plt.axis('off')
            plt.savefig(
                os.path.join(img_prefix, f'epoch-{epoch:04d}.png'),
                bbox_inches='tight',
            )

            plt.close()

            store_state(
                vae,
                discriminator,
                [vae_optimizer, dis_optimizer],
                epoch,
                os.path.join(chkpt_prefix, f'epoch-{epoch:04d}.model'),
            )


def main():
    import argparse as ap

    args = ap.ArgumentParser()

    args.add_argument('data-dir', type=str)

    args.add_argument('--latent-size', type=int, default=100)
    args.add_argument('--spike-prior', action='store_true')
    args.add_argument('--anneal-dkl', action='store_true')

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--report-interval', type=int, default=100)

    opts = vars(args.parse_args())

    data_dir = opts.pop('data-dir')

    from imageio import imread
    image = imread(os.path.join(data_dir, 'image.tif'))
    label = imread(os.path.join(data_dir, 'label.tif'))
    data = pd.read_csv(
        os.path.join(data_dir, 'data.gz'),
        sep=' ',
        header=0,
        index_col=0,
    )

    run(image, label, data, **opts)


if __name__ == '__main__':
    main()
