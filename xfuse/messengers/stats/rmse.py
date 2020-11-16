from .stats_handler import StatsHandler, log_scalar
from ...session import get


class RMSE(StatsHandler):
    r"""Root-mean-square error stats tracker"""

    def _select_msg(self, type, name, is_observed, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and is_observed and name[-3:] == "xsg"

    def _handle(self, fn, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        training_data = get("training_data")
        rmse = ((fn.mean - value) ** 2).mean(1).sqrt().mean().item()
        try:
            training_data.rmse = training_data.rmse + 1e-3 * (
                rmse - training_data.rmse
            )
        except TypeError:
            training_data.rmse = rmse
        log_scalar("accuracy/rmse", rmse)
