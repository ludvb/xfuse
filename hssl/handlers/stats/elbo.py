from .stats_handler import StatsHandler


class ELBO(StatsHandler):
    r"""ELBO stats tracker"""

    def _handle(self, **msg) -> None:
        raise RuntimeError("unreachable code path")

    def _select_msg(self, **msg) -> bool:
        return False

    def _pyro_post_step(self, msg):
        # pylint: disable=no-member
        self.add_scalar(f"loss/elbo", -msg["value"])
