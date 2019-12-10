from abc import abstractmethod, abstractproperty

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
    def num_z(self):
        r"""Number of independent tissue states"""
        return 1

    @abstractproperty
    def tag(self):
        r"""Experiment name"""

    @abstractmethod
    def model(self, x, zs):
        r"""Experiment model"""

    @abstractmethod
    def guide(self, x):
        r"""Experiment guide for :class:`pyro.infer.SVI`"""

    def forward(self, x, zs):  # pylint: disable=arguments-differ
        return self.model(x, zs)
