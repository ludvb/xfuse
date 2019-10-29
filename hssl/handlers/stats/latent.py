from .stats_handler import StatsHandler
from ...utility.visualization import reduce_last_dimension


class Latent(StatsHandler):
    r"""Latent state stats tracker"""

    def _select_msg(self, type, name, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name == "z"

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_images(
            "latent/mean",
            reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
            dataformats="NHWC",
        )
