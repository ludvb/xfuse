from .stats_handler import StatsHandler


class ELBO(StatsHandler):
    r"""ELBO stats tracker"""

    def _handle(self, **msg) -> None:
        # pylint: disable=no-member
        self.add_scalar(f"loss/elbo", -msg["value"])

    def _select_msg(self, type, **msg):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"
