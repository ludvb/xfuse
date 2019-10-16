import os

from typing import Optional

from pyro.poutine.messenger import Messenger

from . import Noop
from ..utility.session import save_session
from ..session import Session, get


class Checkpointer(Messenger):
    def __new__(cls, frequency: Optional[int]):
        if frequency is None:
            return Noop()
        return super().__new__(cls)

    def __init__(self, frequency: int = 1):
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
