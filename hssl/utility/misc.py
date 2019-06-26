import pyro as p

import torch as t


__all__ = [
    'MultivariateNormal',
    'Unpool',
]


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


class MultivariateNormal(p.distributions.MultivariateNormal):
    def log_prob(self, x, *args, **kwargs):
        device = self.loc.device
        self.loc = self.loc.cpu()
        self._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril.cpu()
        result = super().log_prob(x.cpu()).to(device)
        self.loc = self.loc.to(device)
        self._unbroadcasted_scale_tril = \
            self._unbroadcasted_scale_tril.to(device)
        return result
