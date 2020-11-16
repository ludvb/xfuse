from .stats_handler import StatsHandler, log_scalar


class LogLikelihood(StatsHandler):
    r"""Log-likelihood stats tracker"""

    def _select_msg(self, type, is_observed, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and is_observed

    def _handle(self, fn, value, name, scale, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        log_scalar(
            f"loss/loglikelihood/{name}", scale * fn.log_prob(value).sum()
        )
