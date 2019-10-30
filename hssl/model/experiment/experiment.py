from abc import abstractmethod

import torch as t

from ...session import get


class Experiment(t.nn.Module):
    r"""Abstract class defining the experiment type"""

    @property
    def n(self):
        r"""
        Returns the size of the subset of the data corresponding to the `tag`
        of the experiment.
        """
        dataloader = get("dataloader")
        if dataloader is not None:
            return dataloader.dataset.size[self.tag]
        return 0

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
