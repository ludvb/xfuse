from .stats_handler import StatsHandler


class RMSE(StatsHandler):
    def _select_msg(self, name, is_observed, **_):
        return is_observed and name[-3:] == 'xsg'

    def _handle(self, fn, value, **_):
        self.add_scalar(
            'accuracy/rmse',
            ((fn.mean - value) ** 2).mean(1).sqrt().mean(),
        )
