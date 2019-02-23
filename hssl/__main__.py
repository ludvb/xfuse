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
            nf=64,
            num_factors=30,
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
        self._z = t.zeros_like(self.z_mu)

        self.decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 4, 1, 0, bias=True),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1, bias=True),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1, bias=True),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1, bias=True),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1, bias=True),
            # x1
        )

        self.img_mu = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Sigmoid(),
        )
        self.img_sd = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Softplus(),
        )

        self.mixes = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, num_factors, 3, 1, 1, bias=True),
            t.nn.Softmax(dim=1),
        )
        self.scale = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.Conv2d(nf, 1, 3, 1, 1, bias=True),
        )

        self.profiles_mu = t.nn.Parameter(t.zeros(num_factors, num_genes))
        self.profiles_sd = t.nn.Parameter(t.zeros(num_factors, num_genes))
        self._profiles = t.zeros(num_factors, num_genes)

        self.gene_baseline = t.nn.Parameter(t.zeros(num_genes))

        self.logit_mu = t.nn.Parameter(t.zeros(num_genes))
        self.logit_sd = t.nn.Parameter(t.zeros(num_genes))
        self._logit = t.zeros(num_genes)

    @property
    def z(self):
        return self._z

    @property
    def profiles(self):
        return self._profiles

    @property
    def logit(self):
        return self._logit

    def decode(self, z):
        state = self.decoder(z)

        img_mu = self.img_mu(state)
        img_sd = self.img_sd(state)

        mixes = self.mixes(state)
        scale = self.scale(state)

        rate = t.einsum(
            'bfxy,fg,bxy->bgxy',
            mixes,
            t.exp(
                self.gene_baseline.unsqueeze(0)
                + self.profiles
            ),
            t.exp(scale).squeeze(1),
        )
        logit = self.logit

        return dict(
            img_mu=img_mu,
            img_sd=img_sd,
            mixes=mixes,
            scale=scale,
            rate=rate,
            logit=logit,
        )

    def forward(self):
        self._profiles = (
            t.distributions.Normal(
                self.profiles_mu,
                t.nn.functional.softplus(self.profiles_sd),
            )
            .rsample()
        )
        self._logit = (
            t.distributions.Normal(
                self.logit_mu,
                t.nn.functional.softplus(self.logit_sd),
            )
            .rsample()
            .reshape(1, -1, 1, 1)
        )

        self._z = t.distributions.Normal(
            self.z_mu,
            t.nn.functional.softplus(self.z_sd),
        ).rsample()

        return {
            'z': self.z,
            'profiles': self.profiles,
            'logit': self.logit,
            **{
                k: center_crop(v, self._shape)
                for k, v in self.decode(self._z).items()
            },
        }


def store_state(model, optimizers, iteration, file):
    t.save(
        dict(
            model=model.state_dict(),
            optimizers=[x.state_dict() for x in optimizers],
            iteration=iteration,
        ),
        file,
    )


def restore_state(model, optimizers, file):
    state = t.load(file)
    model.load_state_dict(state['vae'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    return state['iteration']


def run(
        image: np.ndarray,
        label: np.ndarray,
        data: pd.DataFrame,
        latent_size: int,
        factors: int,
        output_prefix: str,
        state: dict = None,
        image_interval: int = 50,
        chkpt_interval: int = 10000,
):
    img_prefix = os.path.join(output_prefix, 'images')
    noise_prefix = os.path.join(output_prefix, 'noise')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(output_prefix, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(noise_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    histonet = Histonet(
        image=image,
        data=data,
        latent_size=latent_size,
        num_factors=factors,
    ).to(DEVICE)

    optimizer = t.optim.Adam(
        histonet.parameters(),
        lr=1e-3,
        # betas=(0.5, 0.999),
    )
    if state:
        start_iteration = restore_state(
            histonet,
            [optimizer],
            state,
        )
    else:
        start_iteration = 1

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
    image = image / 255

    label = t.eye(int(label.max()) + 1)[label.flatten()].to(DEVICE)

    fixed_noise = t.distributions.Normal(
        t.zeros([
            int(x * s)
            for x, s in zip(histonet.z_mu.shape, [1, 1, 1, 1])
        ]),
        t.ones([
            int(x * s)
            for x, s in zip(histonet.z_sd.shape, [1, 1, 1, 1])
        ]),
    ).sample().to(DEVICE)

    def _step():
        g = histonet()

        lpimg = (
            t.distributions.Normal(g['img_mu'], g['img_sd'])
            .log_prob(image)
        )

        rates = (
            (g['rate'].reshape(*g['rate'].shape[:2], -1) @ label)
            [:, :, 1:]
            + 1e-10
        )
        # ^ NOTE large memory requirements

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
            logits=g['logit'].reshape(*g['logit'].shape[:2], -1),
        )
        lpobs = d.log_prob(obs)

        dkl = (
            t.sum(
                t.distributions.Normal(
                    histonet.z_mu,
                    t.nn.functional.softplus(histonet.z_sd),
                )
                .log_prob(g['z'])
                -
                t.distributions.Normal(0., 1.).log_prob(g['z'])
            )
            +
            t.sum(
                t.distributions.Normal(
                    histonet.profiles_mu,
                    t.nn.functional.softplus(histonet.profiles_sd),
                )
                .log_prob(g['profiles'])
                -
                t.distributions.Normal(0., 1.).log_prob(g['profiles'])
            )
            +
            t.sum(
                t.distributions.Normal(
                    histonet.logit_mu,
                    t.nn.functional.softplus(histonet.logit_sd),
                )
                .log_prob(g['logit'].squeeze())
                -
                t.distributions.Normal(0., 1.).log_prob(g['logit'].squeeze())
            )
        )

        img_loss = -t.sum(lpimg)
        xpr_loss = -t.sum(lpobs)
        loss = img_loss + xpr_loss + dkl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return collect_items({
            'L': loss,
            'Li': img_loss,
            'Lx': xpr_loss,
            'dqp': dkl,
            'p(img|z)': t.mean(t.sigmoid(lpimg)),
            'p(xpr|z)': t.mean(t.exp(lpobs)),
            'rmse': t.sqrt(t.mean((d.mean - obs) ** 2)),
        })

    t.enable_grad()
    histonet.train()

    def _report(iteration, outputs):
        print(
            f'iteration {iteration:4d}:',
            '  //  '.join([
                '{} = {:>9s}'.format(k, f'{np.mean(v):.2e}')
                for k, v in outputs.items()
            ]),
        )

        with gzip.open(
                os.path.join(output_prefix, 'training_data.csv.gz'),
                'ab',
        ) as fh:
            for k, vs in outputs.items():
                for i, v in enumerate(vs, iteration - len(vs) + 1):
                    fh.write((
                        ','.join([
                            str(i),
                            str(k),
                            str(v),
                        ])
                        + '\n'
                    ).encode())

    def _save_chkpt(iteration):
        store_state(
            histonet,
            [optimizer],
            iteration,
            os.path.join(
                chkpt_prefix,
                f'iteration-{iteration:05d}.pkl',
            ),
        )

    def _save_image(iteration):
        t.no_grad()
        histonet.eval()

        inferred = histonet()
        noise = histonet.decode(fixed_noise)

        for model, d in [
                (inferred, img_prefix),
                (noise, noise_prefix),
        ]:
            imwrite(
                os.path.join(d, f'iteration-{iteration:05d}.jpg'),
                ((
                    t.distributions.Normal(
                        model['img_mu'],
                        model['img_sd'],
                    )
                    .sample()
                    [0]
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                ) * 255).astype(np.uint8),
            )

        t.enable_grad()
        histonet.train()

    last_image = 0.
    last_chkpt = 0.

    for iteration, output in enumerate(
            map(
                lambda x: zip_dicts(filter(lambda y: y is not None, x)),
                it.zip_longest(*[(_step() for _ in it.count())] * 10),
            ),
            start_iteration,
    ):
        subiteration = 10 * iteration

        _report(subiteration, output)

        if subiteration - last_image > image_interval:
            _save_image(subiteration)
            last_image = subiteration

        if subiteration - last_chkpt > chkpt_interval:
            _save_chkpt(subiteration)
            last_chkpt = subiteration


def main():
    import argparse as ap

    args = ap.ArgumentParser()

    args.add_argument('data-dir', type=str)

    args.add_argument('--latent-size', type=int, default=100)
    args.add_argument('--factors', type=int, default=30)

    args.add_argument('--zoom', type=float, default=0.1)
    args.add_argument('--genes', type=int, default=50)

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--image-interval', type=int, default=100)
    args.add_argument('--chkpt-interval', type=int, default=100)

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
        [-opts.pop('genes'):]
        .index
    ]

    print(f'using device: {str(DEVICE):s}')

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

    def run_tsne(y, n_components=3, initial_dims=20):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        x = y.reshape(-1, y.shape[-1])

        def uniformize(x):
            colmax = np.max(x, axis=0)
            colmin = np.min(x, axis=0)
            x = x - colmin.reshape(1, -1)
            ranges = [ma - mi for ma, mi in zip(colmax, colmin)]
            maxrange = max(ranges)
            x = x / maxrange
            return(x)
        # TODO
        # configure initial_dims via CLI
        # set initial_dims intelligently
        # np.savetxt(output_forward_path, x, delimiter='\t') # need to add spot names
        print("performing PCA")
        pca_map = PCA(n_components=initial_dims).fit_transform(x)
        print("performing tSNE")
        tsne_map = TSNE(n_components=n_components).fit_transform(pca_map)
        tsne_map = uniformize(tsne_map)
        # np.savetxt(output_forward_path + ".tsne.tsv", tsne_map, delimiter='\t') # this is 3-d data... need to add spot names
        return tsne_map.reshape((*y.shape[:2], -1))
