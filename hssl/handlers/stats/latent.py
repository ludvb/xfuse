from .stats_handler import StatsHandler
from ...utility.visualization import reduce_last_dimension


class Latent(StatsHandler):
    def _select_msg(self, name, **_):
        return name == "z"

    def _handle(self, fn, **_):
        self.add_images(
            "latent/mean",
            reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
            dataformats="NHWC",
        )
