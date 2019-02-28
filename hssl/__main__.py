# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from datetime import datetime as dt

import gzip

import itertools as it

import os

from imageio import imread

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes

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
                batch,
                nrow=int(np.floor(np.sqrt(len(batch)))),
                padding=5,
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
            num_genes,
            hidden_size=512,
            latent_size=96,
            nf=32,
            gene_bias=None,
    ):
        super().__init__()

        self.encoder = t.nn.Sequential(
            # x1
            t.nn.Conv2d(num_genes + 4, 2 * nf, 4, 2, 2, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * nf),
            # x2
            t.nn.Conv2d(2 * nf, 4 * nf, 4, 2, 2, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * nf),
            # x4
            t.nn.Conv2d(4 * nf, 8 * nf, 4, 2, 2, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * nf),
            # x8
            t.nn.Conv2d(8 * nf, 16 * nf, 4, 2, 2, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            # x16
        )

        self.z_mu = t.nn.Sequential(
            t.nn.Conv2d(16 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, latent_size, 3, 1, 1, bias=True),
        )
        self.z_sd = t.nn.Sequential(
            t.nn.Conv2d(16 * nf, 16 * nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.Conv2d(16 * nf, latent_size, 3, 1, 1, bias=True),
        )

        self.decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(latent_size, 16 * nf, 3, 1, 1, bias=True),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            t.nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1, bias=True),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * nf),
            t.nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1, bias=True),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * nf),
            t.nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1, bias=True),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * nf),
            t.nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1, bias=True),
            # x1
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
        )

        self.img_mu = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Sigmoid(),
        )
        self.img_sd = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, 3, 3, 1, 1, bias=True),
            t.nn.Softplus(),
        )

        self.rate_mu = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, num_genes, 3, 1, 1, bias=True),
        )
        self.rate_sd = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, num_genes, 3, 1, 1, bias=True),
            t.nn.Softplus(),
        )

        self.rate_bias = t.nn.Parameter(
            (
                t.as_tensor(gene_bias).float()
                if gene_bias is not None else
                t.zeros(num_genes)
            ),
            requires_grad=False,
        )

        self.logit_mu = t.nn.Parameter(t.zeros(num_genes, 1))
        self.logit_sd = t.nn.Parameter(-5 * t.ones(num_genes, 1))

    def encode(self, x):
        x = self.encoder(x)
        z_mu = self.z_mu(x)
        z_sd = self.z_sd(x)
        z = t.distributions.Normal(z_mu, z_sd).mean  # rsample()
        return (
            z,
            z_mu,
            z_sd,
        )

    def decode(self, z):
        state = self.decoder(z)

        img_mu = self.img_mu(state)
        img_sd = self.img_sd(state)

        lrate = (
            t.distributions.Normal(
                self.rate_mu(state),
                t.nn.functional.softplus(self.rate_sd(state)),
            )
            # .rsample()
            .mean
        )

        rate = t.exp(self.rate_bias[None, ..., None, None] + lrate)

        logit = (
            t.distributions.Normal(
                self.logit_mu,
                t.nn.functional.softplus(self.logit_sd),
            )
            # .rsample()
            .mean
        )

        return (
            img_mu,
            img_sd,
            rate,
            logit,
        )

    def forward(self, x):
        shape = x.shape[-2:]
        z, z_mu, z_sd = self.encode(x)
        ys = self.decode(z)
        return (
            z,
            *[
                center_crop(y, [None, None, *shape])
                if len(y.shape) == 4 else
                y
                for y in ys
            ],
        )


def store_state(model, optimizers, epoch, file):
    t.save(
        dict(
            model=model.state_dict(),
            optimizers=[x.state_dict() for x in optimizers],
            epoch=epoch,
        ),
        file,
    )


def restore_state(model, optimizers, file):
    state = t.load(file)
    model.load_state_dict(state['model'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    return state['epoch']


class Dataset(t.utils.data.Dataset):
    def __init__(
            self,
            image: t.Tensor,
            label: np.ndarray,
            data: pd.DataFrame,
            patch_size: int = 700,
    ):
        self.image = image
        self.label = label

        self.data = t.tensor(data.values).float()
        self.data = t.cat([t.zeros(self.data.shape[0])[:, None], self.data], 1)
        self.data = t.cat([t.zeros(self.data.shape[1])[None, :], self.data], 0)
        self.data[0, 0] = 1.

        self.h, self.w = [min(s, patch_size) for s in image.shape[-2:]]

    def __len__(self):
        return int(np.ceil(
            np.product(self.image.shape[-2:]) / self.h / self.w))

    def __getitem__(self, idx):
        y, x = [
            np.random.randint(s - d + 1)
            for s, d in zip(self.image.shape[-2:], (self.h, self.w))
        ]

        image = self.image[:, y:y + self.h, x:x + self.w].clone()
        label = self.label[y:y + self.h, x:x + self.w].copy()

        # remove partially visible labels
        label[np.invert(binary_fill_holes(label == 0))] = 0

        labels = [*sorted(np.unique(label))]
        data = self.data[labels, :]
        label = t.tensor(np.searchsorted(labels, label))

        return dict(
            image=image,
            label=label,
            data=data,
        )


def run(
        image: np.ndarray,
        label: np.ndarray,
        data: pd.DataFrame,
        latent_size: int,
        output_prefix: str,
        patch_size: int = 700,
        batch_size: int = 5,
        state: dict = None,
        image_interval: int = 50,
        chkpt_interval: int = 10000,
        workers: int = None,
):
    if workers is None:
        workers = len(os.sched_getaffinity(0))

    img_prefix = os.path.join(output_prefix, 'images')
    noise_prefix = os.path.join(output_prefix, 'noise')
    chkpt_prefix = os.path.join(output_prefix, 'checkpoints')

    os.makedirs(output_prefix, exist_ok=True)
    os.makedirs(img_prefix, exist_ok=True)
    os.makedirs(noise_prefix, exist_ok=True)
    os.makedirs(chkpt_prefix, exist_ok=True)

    histonet = Histonet(
        num_genes=len(data.columns),
        latent_size=latent_size,
        gene_bias=np.log(
            (data.values / np.bincount(label.flatten())[1:][..., None])
            .mean(0)
        ),
    ).to(DEVICE)

    optimizer = t.optim.Adam(
        histonet.parameters(),
        lr=1e-5,
        # betas=(0.5, 0.999),
    )
    if state:
        start_epoch = restore_state(
            histonet,
            [optimizer],
            state,
        )
    else:
        start_epoch = 1

    image = t.tensor(image).permute(2, 0, 1).float() / 255

    dataset = Dataset(image, label, data, patch_size=patch_size)

    def _collate(xs):
        data = [x.pop('data') for x in xs]
        nlabels = [len(d) - 1 for d in data]
        labels = [
            (l + n) * (l != 0).long() for l, n in
            zip((x.pop('label') for x in xs), it.accumulate((0, *nlabels)))
        ]
        return dict(
            data=t.cat([data[0], *(d[1:] for d in data[1:])]),
            label=t.stack(labels),
            **{k: t.stack([x[k] for x in xs]) for k in xs[0].keys()},
        )

    dataloader = t.utils.data.DataLoader(
        dataset,
        collate_fn=_collate,
        batch_size=batch_size,
        num_workers=workers,
    )

    fixed_data = next(iter(dataloader))

    # fixed_noise = t.distributions.Normal(
    #     t.zeros([
    #         int(x * s)
    #         for x, s in zip(histonet.z_mu.shape, [1, 1, 1, 1])
    #     ]),
    #     t.ones([
    #         int(x * s)
    #         for x, s in zip(histonet.z_sd.shape, [1, 1, 1, 1])
    #     ]),
    # ).sample().to(DEVICE)

    # np.random.seed(1337)
    # num_labels = label.max()
    # vset = np.random.choice(num_labels, int(0.2 * num_labels))
    # tset = np.setdiff1d(range(num_labels), vset)

    def _run_histonet_on(x):
        spatial_data = (
            x['data']
            [x['label'].flatten()]
            .reshape(*x['label'].shape, -1)
            .permute(0, 3, 1, 2)
        )

        return histonet(t.cat([x['image'], spatial_data], 1))

    def _step(x):
        x = {k: v.to(DEVICE) for k, v in x.items()}

        z, img_mu, img_sd, rate, logit = _run_histonet_on(x)

        lpimg = (
            t.distributions.Normal(img_mu, img_sd)
            .log_prob(x['image'])
        )

        label = (
            t.eye(t.max(x['label']) + 1, device=DEVICE)
            [x['label'].flatten()]
            .reshape(*x['label'].shape, -1)
            .float()
        )
        rates = t.einsum('bgxy,bxyi->ig', rate, label)

        d = t.distributions.NegativeBinomial(
            rates[1:] + 1e-10,
            logits=logit.t(),
        )

        data = x['data'][1:, 1:]
        lpobs = d.log_prob(data)

        dkl = t.tensor(0.)
        # (
        #     t.sum(
        #         t.distributions.Normal(
        #             histonet.z_mu,
        #             t.nn.functional.softplus(histonet.z_sd),
        #         )
        #         .log_prob(z)
        #         -
        #         t.distributions.Normal(0., 1.).log_prob(z)
        #     )
        #     +
        #     t.sum(
        #         t.distributions.Normal(
        #             histonet.logit_mu,
        #             t.nn.functional.softplus(histonet.logit_sd),
        #         )
        #         .log_prob(logit)
        #         -
        #         t.distributions.Normal(0., 1.).log_prob(logit)
        #     )
        # )

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
            'p(img|z)': t.mean(t.exp(lpimg)),
            'p(xpr|z)': t.mean(t.exp(lpobs)),
            'rmse': t.sqrt(t.mean((d.mean - data) ** 2)),
        })

    t.enable_grad()
    histonet.train()

    def _report(epoch, output):
        iteration = output.pop('iteration')

        print(
            f'epoch {epoch:4d}',
            ' '
            f'(%0{len(str(len(dataloader)))}d/{len(dataloader)})'
            % iteration[-1],
            ' :: ',
            '  //  '.join([
                '{} = {:>9s}'.format(k, f'{np.mean(v):.2e}')
                for k, v in output.items()
            ]),
        )

        with gzip.open(
                os.path.join(output_prefix, 'training_data.csv.gz'),
                'ab',
        ) as fh:
            for k, vs in output.items():
                for i, v in zip(iteration, vs):
                    fh.write((
                        ','.join([
                            str(epoch),
                            str(i),
                            str(k),
                            str(v),
                        ])
                        + '\n'
                    ).encode())

    def _save_chkpt(epoch):
        store_state(
            histonet,
            [optimizer],
            epoch,
            os.path.join(
                chkpt_prefix,
                f'epoch-{epoch:05d}.pkl',
            ),
        )

    def _save_image(epoch):
        t.no_grad()
        histonet.eval()

        _, imu, isd, *_ = _run_histonet_on({
            k: v.to(DEVICE) for k, v in fixed_data.items()
        })
        # _, nmu, nsd, *_ = histonet.decode(fixed_noise)

        for (mu, sd), d in [
                ((imu, isd), img_prefix),
                # ((nmu, nsd), noise_prefix),
        ]:
            visualize_batch(
                t.distributions.Normal(mu, sd)
                .sample()
                .clamp(0, 1)
                .detach()
                .cpu()
            )
            plt.savefig(os.path.join(d, f'epoch-{epoch:05d}.jpg'))

        t.enable_grad()
        histonet.train()

    for epoch in it.count(start_epoch):
        for output in map(
                lambda x: zip_dicts(
                    map(
                        lambda y: {'iteration': y[0], **y[1]},
                        filter(lambda y: y is not None, x),
                    ),
                ),
                it.zip_longest(
                    *[(
                        (i, _step(x)) for i, x in enumerate(dataloader, 1)
                    )] * 10
                ),
        ):
            _report(epoch, output)

        if epoch % image_interval == 0:
            _save_image(epoch)
        if epoch % chkpt_interval == 0:
            _save_chkpt(epoch)


def main():
    import argparse as ap

    args = ap.ArgumentParser()

    args.add_argument('data-dir', type=str)

    args.add_argument('--latent-size', type=int, default=100)

    args.add_argument('--zoom', type=float, default=1.)
    args.add_argument('--genes', type=int, default=50)
    args.add_argument('--patch-size', type=int, default=700)
    args.add_argument('--batch-size', type=int, default=5)

    args.add_argument(
        '--output-prefix',
        type=str,
        default=f'./hssl-{dt.now().isoformat()}',
    )
    args.add_argument('--state', type=str)
    args.add_argument('--image-interval', type=int, default=100)
    args.add_argument('--chkpt-interval', type=int, default=100)
    args.add_argument('--workers', type=int)

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

    zoom_level = opts.pop('zoom')
    if zoom_level < 1:
        from scipy.ndimage.interpolation import zoom
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
    from scipy.ndimage.interpolation import zoom
    num_genes = 50
    data = pd.read_csv('~/histonet-test-data/data.gz', sep=' ').set_index('n')
    data = data[
        data.sum(0)[[
            x for x in data.columns if 'ambiguous' not in x
        ]]
        .sort_values()
        [-num_genes:]
        .index
    ]
    label = imread('~/histonet-test-data/label.tif')
    image = imread('~/histonet-test-data/image.tif')
    label = zoom(label, (0.1, 0.1), order=0)
    image = zoom(image, (0.1, 0.1, 1))

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
        print("performing PCA")
        pca_map = PCA(n_components=initial_dims).fit_transform(x)
        print("performing tSNE")
        tsne_map = TSNE(n_components=n_components).fit_transform(pca_map)
        tsne_map = uniformize(tsne_map)
        return tsne_map.reshape((*y.shape[:2], -1))

    def visualize(model, z):
        _, nmu, nsd, *_ = model.decode(z)
        return plt.imshow(nmu[0].detach().numpy().transpose(1, 2, 0))

    def interpolate(z1, z2, to='/tmp/interpolation.mp4'):
        from matplotlib.animation import ArtistAnimation
        fig = plt.figure()
        anim = ArtistAnimation(
            fig,
            [[visualize(z1 + (z2 - z1) * k)] for k in np.linspace(0, 1, 100)],
            repeat_delay=1000,
            interval=50,
            blit=True,
        )
        anim.save(to)
