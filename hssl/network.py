import torch as t

from .distributions import Distribution, Normal, Variable
from .logging import DEBUG, log
from .utility import center_crop


class Variational(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latents = dict()

    def _register_latent(
            self,
            variable: Variable,
            prior: Distribution,
            is_global: bool = False,
    ):
        varid = id(variable)
        if varid in self._latents:
            raise RuntimeError(f'variable {id} has already been registered')

        log(DEBUG, 'registering latent variable %d', varid)
        self._latents[varid] = variable, prior, is_global

    def complexity_cost(self, batch_fraction):
        return sum([
            t.sum(x.distribution.log_prob(x.value) - p.log_prob(x.value))
            * (batch_fraction if g else 1.)
            for x, p, g in self._latents.values()
        ])


class Unpool(t.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=3,
            stride=2,
            padding=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if padding is None:
            padding = kernel_size // 2

        self.conv = t.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.scale_factor = stride

    def forward(self, x):
        x = t.nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(x)
        return x


class Histonet(Variational):
    def __init__(
            self,
            num_genes,
            hidden_size=512,
            latent_size=96,
            nf=32,
            gene_bias=None,
    ):
        self._init_args = locals()
        self._init_args.pop('__class__')
        self._init_args.pop('self')

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
        self.z = Variable(Normal())
        self._register_latent(self.z, Normal())

        self.decoder = t.nn.Sequential(
            t.nn.Conv2d(latent_size, 16 * nf, 5, padding=2),
            # x16
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(16 * nf),
            Unpool(16 * nf, 8 * nf, 5),
            # x8
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(8 * nf),
            Unpool(8 * nf, 4 * nf, 5),
            # x4
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(4 * nf),
            Unpool(4 * nf, 2 * nf, 5),
            # x2
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(2 * nf),
            Unpool(2 * nf, nf, 5),
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

        self.rate = t.nn.Sequential(
            t.nn.Conv2d(nf, nf, 3, 1, 2, bias=True),
            t.nn.LeakyReLU(0.2, inplace=True),
            t.nn.BatchNorm2d(nf),
            t.nn.Conv2d(nf, num_genes, 3, 1, 2, bias=True),
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
        self.logit = Variable(Normal())
        self._register_latent(self.logit, Normal(), is_global=True)

    @property
    def init_args(self):
        return self._init_args

    def encode(self, x):
        x = self.encoder(x)
        z_mu = self.z_mu(x)
        z_sd = self.z_sd(x)

        self.z.distribution.set(loc=z_mu, scale=z_sd, r_transform=True)
        z = self.z.sample().value

        return (
            z,
            z_mu,
            z_sd,
        )

    def decode(self, z):
        state = self.decoder(z)

        img_mu = self.img_mu(state)
        img_sd = self.img_sd(state)

        rate = t.exp(self.rate_bias[None, ..., None, None] + self.rate(state))

        self.logit.distribution.set(
            loc=self.logit_mu, scale=self.logit_sd, r_transform=True)
        logit = self.logit.sample().value

        return (
            img_mu,
            img_sd,
            rate,
            logit,
            state,
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
