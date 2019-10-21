from abc import abstractmethod

import torch as t


class Experiment(t.nn.Module):
    r"""Abstract class defining the experiment type"""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    @property
    @abstractmethod
    def tag(self):
        r"""Experiment name"""

    @abstractmethod
    def model(self, x, z):
        r"""Experiment model"""

    @abstractmethod
    def guide(self, x):
        r"""Experiment guide for :class:`pyro.infer.SVI`"""

    def forward(self, x, z):  # pylint: disable=arguments-differ
        return self.model(x, z)
