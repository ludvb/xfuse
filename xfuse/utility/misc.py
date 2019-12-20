import torch

__all__ = ["Unpool"]


class Unpool(torch.nn.Module):
    r"""
    A :class:`~torch.nn.Module` combining upsampling with a convolutional layer
    """

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

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.scale_factor = stride

    def forward(self, x):
        # pylint: disable=arguments-differ
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(x)
        return x
