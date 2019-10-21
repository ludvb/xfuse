from .stats_handler import StatsHandler


class RMSE(StatsHandler):
    r"""Root-mean-square error stats tracker"""

    def _select_msg(self, name, is_observed, **_):
        # pylint: disable=arguments-differ
        return is_observed and name[-3:] == "xsg"

    def _handle(self, fn, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_scalar(
            "accuracy/rmse", ((fn.mean - value) ** 2).mean(1).sqrt().mean()
        )
