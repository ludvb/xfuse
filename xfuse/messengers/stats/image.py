from .stats_handler import StatsHandler, log_images


class Image(StatsHandler):
    r"""Image stats tracker"""

    def _select_msg(self, type, name, is_observed, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and is_observed and name[-5:] == "image"

    def _handle(self, fn, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        log_images("image/ground_truth", (1 + value.permute(0, 2, 3, 1)) / 2)
        log_images("image/mean", (1 + fn.mean.permute(0, 2, 3, 1)) / 2)
        log_images("image/sample", (1 + fn.sample().permute(0, 2, 3, 1)) / 2)
