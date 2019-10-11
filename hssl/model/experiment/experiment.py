from abc import abstractmethod

import pyro as p

import torch as t


class Experiment(t.nn.Module):
    def __init__(self, n: int):
        self.n = n

    def _sample_global(name, *args, **kwargs):
        try:
            return p.sample(name, *args, **kwargs)
        except RuntimeError:
            return p.poutine.runtime._PYRO_STACK[-1].trace.nodes[name]["value"]

    @property
    @abstractmethod
    def tag(self):
        pass

    @abstractmethod
    def model(self, x, z):
        pass

    @abstractmethod
    def guide(self, x):
        pass
