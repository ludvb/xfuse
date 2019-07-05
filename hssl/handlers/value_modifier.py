from pyro.poutine.messenger import Messenger


class ValueModifier(Messenger):
    def __init__(self, sites, modifier):
        self._sites = sites
        self._modifier = modifier

    def _postprocess_message(self, msg):
        if msg['name'] in self._sites and msg['value'] is not None:
            msg['value'] = self._modifier(msg['value'])
