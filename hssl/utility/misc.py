import torch

__all__ = ["PaddedConv2d", "Unpool"]


class PaddedConv2d(torch.nn.Module):
    def __init__(
        self, *args, padding: int = 0, padding_mode: str = "reflect", **kwargs
    ):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(*args, padding=0, **kwargs)
        self.padding_mode = padding_mode
        self.padding = padding

    def forward(self, x):
        x = torch.nn.functional.pad(
            x,
            [self.padding, self.padding, self.padding, self.padding],
            mode=self.padding_mode,
        )
        return self.conv_layer(x)


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
