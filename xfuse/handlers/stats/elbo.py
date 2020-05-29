from .stats_handler import StatsHandler
from ...session import get


class ELBO(StatsHandler):
    r"""ELBO stats tracker"""

    def _handle(self, **msg) -> None:
        # pylint: disable=no-member
        training_data = get("training_data")
        try:
            training_data.elbo_short = training_data.elbo_short + 1e-3 * (
                msg["value"] - training_data.elbo_short
            )
            training_data.elbo_long = training_data.elbo_long + 1e-4 * (
                msg["value"] - training_data.elbo_long
            )
        except TypeError:
            training_data.elbo_short = msg["value"]
            training_data.elbo_long = msg["value"]
        self.add_scalar("loss/elbo", msg["value"])

    def _select_msg(self, type, **msg):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"
