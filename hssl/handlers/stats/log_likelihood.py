from .stats_handler import StatsHandler


class LogLikelihood(StatsHandler):
    r"""Log-likelihood stats tracker"""

    def _select_msg(self, is_observed, **_):
        # pylint: disable=arguments-differ
        return is_observed

    def _handle(self, fn, value, name, scale, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_scalar(
            f"loss/loglikelihood/{name}", scale * fn.log_prob(value).sum()
        )
