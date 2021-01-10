import re

import torch

from .stats_handler import StatsHandler, log_scalar
from ...session import get


class RMSE(StatsHandler):
    r"""Root-mean-square error stats tracker"""

    def _select_msg(self, type, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"

    def _handle(self, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        training_data = get("training_data")

        try:
            means, values = zip(
                *[
                    (x["fn"].mean, x["value"])
                    for x in value.nodes.values()
                    if re.match("ST/xsg-\\d+", x["site"])
                ]
            )
        except ValueError:
            return

        rmse = (
            ((torch.cat(means) - torch.cat(values)) ** 2)
            .mean(1)
            .sqrt()
            .mean()
            .item()
        )

        try:
            training_data.rmse = training_data.rmse + 1e-3 * (
                rmse - training_data.rmse
            )
        except TypeError:
            training_data.rmse = rmse

        log_scalar("accuracy/rmse", rmse)
