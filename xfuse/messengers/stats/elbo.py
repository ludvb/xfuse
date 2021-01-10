from .stats_handler import StatsHandler, log_scalar
from ...session import get


class ELBO(StatsHandler):
    r"""ELBO stats tracker"""

    def _select_msg(self, type, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"

    def _handle(self, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        training_data = get("training_data")
        model_log_prob = value.log_prob_sum(
            site_filter=lambda _, x: x["is_guide"]
        )
        guide_log_prob = value.log_prob_sum(
            site_filter=lambda _, x: not x["is_guide"]
        )
        elbo = (guide_log_prob - model_log_prob).item()
        try:
            training_data.elbo_short = training_data.elbo_short + 1e-3 * (
                elbo - training_data.elbo_short
            )
            training_data.elbo_long = training_data.elbo_long + 1e-4 * (
                elbo - training_data.elbo_long
            )
        except TypeError:
            training_data.elbo_short = elbo
            training_data.elbo_long = elbo
        log_scalar("loss/elbo", elbo)
