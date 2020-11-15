import os

from pyro.poutine.messenger import Messenger

from ..session.io import save_session


class Checkpointer(Messenger):
    r"""Saves the currently running session to disk at a fixed interval"""

    def __init__(self, period: int = 1):
        super().__init__()
        self._period = period

    def _pyro_post_epoch(self, msg):
        epoch = msg["kwargs"]["epoch"]
        if epoch % self._period == 0:
            save_session(os.path.join("checkpoints", f"epoch-{epoch:08d}"))
