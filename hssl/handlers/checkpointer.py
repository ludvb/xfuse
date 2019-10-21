import os
from typing import Optional

from pyro.poutine.messenger import Messenger

from ..session import Session, get
from ..utility.session import save_session
from . import Noop


class Checkpointer(Messenger):
    r"""Saves the currently running session to disk at a fixed interval"""

    def __new__(cls, frequency: Optional[int]):
        if frequency is None:
            return Noop()
        return super().__new__(cls)

    def __init__(self, frequency: int = 1):
        super().__init__()
        self._freq = frequency

    def _pyro_post_epoch(self, msg):
        if msg["kwargs"]["epoch"] % self._freq == 0:
            with Session(pyro_stack=[]):
                save_session(
                    os.path.join(
                        get("save_path"),
                        "checkpoints",
                        f"{int(get('global_step')):08d}.pkl",
                    )
                )
