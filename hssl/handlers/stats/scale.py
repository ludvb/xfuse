from .stats_handler import StatsHandler


class Scale(StatsHandler):
    r"""Scaling factor stats tracker"""

    def _select_msg(self, name, **_):
        # pylint: disable=arguments-differ
        return name[-5:] == "scale"

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_images(
            "scale", fn.mean.permute(0, 2, 3, 1), dataformats="NHWC"
        )
