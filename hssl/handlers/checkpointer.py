import os

from pyro.poutine.messenger import Messenger

from ..utility.session import save_session
from ..session import Session, get_global_step, get_save_path


class Checkpointer(Messenger):
    def __init__(self, frequency: int = 1):
        self._freq = frequency

    def _pyro_post_epoch(self, msg):
        if msg['kwargs']['epoch'] % self._freq == 0:
            with Session(pyro_stack=[]):
                save_session(os.path.join(
                    get_save_path(),
                    'checkpoints',
                    f'{int(get_global_step()):08d}.pkl',
                ))
