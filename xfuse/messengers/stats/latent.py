import torch

from .stats_handler import StatsHandler, log_images
from ...utility.visualization import reduce_last_dimension


class Latent(StatsHandler):
    r"""Latent state stats tracker"""

    def _select_msg(self, type, name, **msg):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name[:2] == "z-" and not msg["is_guide"]

    def _handle(self, value, name, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        try:
            log_images(
                f"z/{name[2:]}",
                torch.as_tensor(
                    reduce_last_dimension(value.permute(0, 2, 3, 1))
                ),
            )
        except ValueError:
            pass
