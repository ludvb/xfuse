from .stats_handler import StatsHandler


class Image(StatsHandler):
    r"""Image stats tracker"""

    def _select_msg(self, type, name, is_observed, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and is_observed and name[-5:] == "image"

    def _handle(self, fn, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_images("image/ground_truth", (1 + value) / 2)
        self.add_images("image/mean", (1 + fn.mean) / 2)
        self.add_images("image/sample", (1 + fn.sample()) / 2)
