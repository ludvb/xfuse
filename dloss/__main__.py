# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import gzip

import itertools as it

import os

import matplotlib.pyplot as plt

import numpy as np

import torch as t
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as tvt
from torchvision.utils import make_grid


DATA_DIR = '~/.local/data'
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


class VAE(t.nn.Module):
    def __init__(
            self,
            hidden_size=512,
            latent_size=96,
    ):
        super().__init__()
        self.activation = t.nn.Softplus()

        self.enc1 = t.nn.Sequential(
            t.nn.Linear(794, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
        )
        self.enc2 = t.nn.Sequential(
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
        )

        self.z_mu = t.nn.Linear(hidden_size, latent_size)
        self.z_sd = t.nn.Linear(hidden_size, latent_size)

        self.dec11 = t.nn.Sequential(
            t.nn.Linear(latent_size, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
        )
        self.dec12 = t.nn.Sequential(
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
        )
        self.dec13 = t.nn.Linear(hidden_size, 784)

        self.dec21 = t.nn.Linear(latent_size, hidden_size)
        self.dec22 = t.nn.Linear(hidden_size, hidden_size)
        self.dec23 = t.nn.Linear(hidden_size, 10)

        self.softmax = t.nn.LogSoftmax(dim=1)

    def encode(self, x, label):
        x = x.reshape(x.shape[0], -1)
        x = t.cat([x, t.eye(10).cuda()[label]], 1)

        x = self.activation(self.enc1(x))
        x = self.activation(self.enc2(x))

        z_mu = self.z_mu(x)
        z_sd = t.nn.functional.softplus(self.z_sd(x))

        return z_mu, z_sd

    def decode(self, z):
        y1 = self.activation(self.dec11(z))
        y1 = self.activation(self.dec12(y1))
        y1 = t.sigmoid(self.dec13(y1))
        y1 = y1.reshape(y1.shape[0], 1, 28, 28)

        y2 = self.activation(self.dec21(z))
        y2 = self.activation(self.dec22(y2))
        y2 = self.activation(self.dec23(y2))
        y2 = self.softmax(y2)

        return y1, y2

    def forward(self, x, label):
        z_mu, z_sd = self.encode(x, label)
        z = t.distributions.Normal(z_mu, z_sd).rsample()
        y1, y2 = self.decode(z)
        return z, z_mu, z_sd, y1, y2


class Discriminator(t.nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super().__init__()
        self.activation = t.nn.Softplus()
        self.fc1 = t.nn.Linear(input_size, hidden_size)
        self.fc2 = t.nn.Linear(hidden_size, hidden_size)
        self.prediction = t.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.prediction(x)
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
        latent_size,
        output_prefix,
        state=None,
        report_interval=50,
):
    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(output_prefix, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    dataset = MNIST(
        DATA_DIR,
        download=True,
        transform=tvt.Compose([
            tvt.ToTensor(),
        ]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=len(os.sched_getaffinity(0)),
    )

    vae = t.nn.DataParallel(
        VAE(latent_size=latent_size).to(DEVICE))
    discriminator = t.nn.DataParallel(
        Discriminator(28 * 28 + latent_size).to(DEVICE))

    vae_optimizer = t.optim.Adam(
        vae.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    dis_optimizer = t.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
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

    for epoch in it.count(start_epoch):

        def _step(x, label):
            x = x.to(DEVICE)
            label = label.to(DEVICE)

            z, z_mu, z_sd, y1, y2 = vae(x, label)

            label_likelihood = -t.nn.functional.nll_loss(
                y2, label, reduction='sum')

            yz = t.cat([y1.reshape(y1.shape[0], -1), z], dim=1)

            pimg1 = t.nn.functional.logsigmoid(discriminator(yz))

            dkl = t.sum(
                t.distributions.kl_divergence(
                    t.distributions.Normal(z_mu, z_sd),
                    t.distributions.Normal(0., 1.),
                ))

            # generator_loss = -(label_likelihood + t.sum(pimg1)) + dkl
            generator_loss = -(label_likelihood + t.sum(pimg1)) + 0.1 * dkl

            vae_optimizer.zero_grad()
            generator_loss.backward()
            vae_optimizer.step()

            pimg2 = discriminator(yz.detach())
            preal = discriminator(
                t.cat([x.reshape(x.shape[0], -1), z.detach()], 1))

            discriminator_loss = (
                - t.sum(t.nn.functional.logsigmoid(1 - pimg2))
                - t.sum(t.nn.functional.logsigmoid(preal))
            )

            dis_optimizer.zero_grad()
            discriminator_loss.backward()
            dis_optimizer.step()

            return (
                len(x),
                collect_items({
                    'loss':
                    generator_loss + discriminator_loss,
                    'd-loss':
                    discriminator_loss,
                    'g-loss':
                    generator_loss,
                    'p(lab|z)':
                    label_likelihood,
                    'p(img|z)':
                    t.mean(t.exp(pimg1)),
                    'dqp':
                    dkl,
                    'accuracy':
                    t.mean(((t.argmax(y2, 1)) == label).float()),
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
                '{} = {:>11s}'.format(k, f'{v:.4e}')
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

            imgs, logits = vae.module.decode(
                t.randn((64, latent_size)).to(DEVICE))

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

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./dloss-{dt.now().isoformat()}',
    )
    args.add_argument('--latent-size', type=int, default=256)
    args.add_argument('--state', type=str)
    args.add_argument('--report-interval', type=int, default=100)

    opts = vars(args.parse_args())

    run(**opts)


if __name__ == '__main__':
    main()
