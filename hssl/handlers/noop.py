from pyro.poutine.messenger import Messenger


class Noop(Messenger):
    r"""No-op :class:`Messenger` that ignores all messages"""

    def _process_message(self, msg):
        pass

    def _postprocess_message(self, msg):
        pass
