from copy import copy

import pyro


class TraceWithDuplicates(pyro.poutine.trace_messenger.TraceMessenger):
    """
    A version of :class:`pyro.poutine.trace_messenger.TraceMessenger` that
    allows tracing with duplicated sample sites. This is necessary for tracing
    the guide and model simultaneously.
    """

    def _pyro_post_sample(self, msg):
        msg = copy(msg)
        msg["site"] = msg.pop("name")
        msg["name"] = str(len(self.trace.nodes))
        super()._pyro_post_sample(msg)

    def _pyro_post_param(self, msg):
        msg = copy(msg)
        msg["site"] = msg.pop("name")
        msg["name"] = str(len(self.trace.nodes))
        super()._pyro_post_param(msg)
