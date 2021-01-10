import re

import torch

from .stats_handler import StatsHandler, log_scalars
from ...session import get


class Conditions(StatsHandler):
    r"""Root-mean-square error stats tracker"""

    def _select_msg(self, type, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"

    def _handle(self, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        for slide, covariate, logits in [
            (*match.groups(), x["value"])
            for x in value.nodes.values()
            for match in [re.match("^logits-(.*)-(.*)$", x["site"])]
            if match
        ]:
            log_scalars(
                "/".join(["conditions", slide, covariate]),
                {
                    condition: prob.item()
                    for prob, condition in zip(
                        torch.softmax(logits, 0), get("covariates")[covariate]
                    )
                },
            )
