from pyro.poutine.messenger import Messenger


class Noop(Messenger):
    def _process_message(self, msg):
        pass

    def _postprocess_message(self, msg):
        pass
