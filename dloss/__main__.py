# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import gzip

import itertools as it

import os

import matplotlib.pyplot as plt

import numpy as np

import torch as t
from torch.utils.data import DataLoader, Dataset

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
        self.activation = t.nn.LeakyReLU(inplace=True)

        self.encoder = t.nn.Sequential(
            t.nn.Linear(795, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.BatchNorm1d(hidden_size),
            t.nn.LeakyReLU(0.2, True),
        )

        self.z_mu = t.nn.Linear(hidden_size, latent_size)
        self.z_sd = t.nn.Linear(hidden_size, latent_size)

        self.image_decoder = t.nn.Sequential(
            t.nn.Linear(latent_size, 256),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(256, 512),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(512, 1024),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(1024, 784),
            t.nn.Tanh(),
        )

        self.label_decoder = t.nn.Sequential(
            t.nn.Linear(latent_size, hidden_size),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(hidden_size, 10),
        )

        self.softmax = t.nn.LogSoftmax(dim=1)

    def encode(self, x, label):
        x = x.reshape(x.shape[0], -1)
        x = t.cat([x, t.eye(11)[label].to(DEVICE)], 1)

        x = self.encoder(x)

        z_mu = self.z_mu(x)
        z_sd = t.nn.functional.softplus(self.z_sd(x))

        return z_mu, z_sd

    def decode(self, z):
        y1 = self.image_decoder(z)
        y1 = y1.reshape(y1.shape[0], 1, 28, 28)

        y2 = self.label_decoder(z)
        y2 = self.softmax(y2)

        return y1, y2

    def forward(self, x, label):
        z_mu, z_sd = self.encode(x, label)
        z = t.distributions.Normal(z_mu, z_sd).rsample()
        y1, y2 = self.decode(z)
        return z, z_mu, z_sd, y1, y2


class Discriminator(t.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.discriminator = t.nn.Sequential(
            t.nn.Linear(input_size, 1024),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(1024, 512),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(512, 256),
            t.nn.LeakyReLU(0.2, True),
            t.nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
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


class PartiallyObservedMNIST(Dataset):
    def __init__(self, obsprop):
        self.dataset = MNIST(
            DATA_DIR,
            download=True,
            transform=tvt.Compose([
                tvt.ToTensor(),
                tvt.Normalize((0.5, ), (0.5, )),
            ]),
        )
        self.observed = t.rand(len(self)) < obsprop
        print(f'datset size = {len(self)}, '
              f'observed = {self.observed.sum()}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, label if self.observed[idx] else t.tensor(10).long()


def run(
        latent_size,
        output_prefix,
        state=None,
        report_interval=50,
        observed=1.0,
        spike_prior=False,
):
    img_prefix = os.path.join(output_prefix, 'images')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(output_prefix, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    dataset = PartiallyObservedMNIST(observed)
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
        lr=1e-3,
        betas=(0.5, 0.999),
        weight_decay=1e-5,
    )
    dis_optimizer = t.optim.Adam(
        discriminator.parameters(),
        lr=1e-3,
        betas=(0.5, 0.999),
        weight_decay=1e-5,
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

        def _step(x, label, observed_label):
            x = x.to(DEVICE)
            label = label.to(DEVICE)
            observed_label = observed_label.to(DEVICE)

            z, z_mu, z_sd, y1, y2 = vae(x, observed_label)

            yz = t.cat([y1.reshape(y1.shape[0], -1), z], dim=1)

            # -* discriminator loss *-
            limg1 = discriminator(yz.detach())
            preal = discriminator(
                t.cat([x.reshape(x.shape[0], -1), z.detach()], 1))

            discriminator_loss = (
                - t.sum(t.nn.functional.logsigmoid(-limg1))
                - t.sum(t.nn.functional.logsigmoid(preal))
            )

            dis_optimizer.zero_grad()
            discriminator_loss.backward()
            dis_optimizer.step()

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
                + dkl * t.tanh(t.as_tensor(epoch).float().to(DEVICE) / 100)
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
                    'unobs. acc':
                    t.mean(correct.masked_select(
                        observed_label == 10).float()),
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

    args.add_argument('--observed', type=float, default=1.0)
    args.add_argument('--latent-size', type=int, default=256)
    args.add_argument('--spike-prior', action='store_true')

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./dloss-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--report-interval', type=int, default=100)

    opts = vars(args.parse_args())

    run(**opts)


if __name__ == '__main__':
    main()
