from .stats_handler import StatsHandler, log_images


class Scale(StatsHandler):
    r"""Scaling factor stats tracker"""

    def _select_msg(self, type, name, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name[-5:] == "scale"

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        scale = fn.mean.permute(0, 2, 3, 1)
        scale = scale / scale.max()
        log_images("scale", scale)
