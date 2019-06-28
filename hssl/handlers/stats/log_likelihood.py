from .stats_handler import StatsHandler


class LogLikelihood(StatsHandler):
    def _select_msg(self, is_observed, **_):
        return is_observed

    def _handle(self, fn, value, name, **_):
        self.add_scalar(f'loss/loglikelihood/{name}', fn.log_prob(value).sum())
