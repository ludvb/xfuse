from abc import abstractmethod, abstractproperty
from typing import Optional

import torch


class Experiment(torch.nn.Module):
    r"""Abstract class defining the experiment type"""

    def __init__(self, size: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._size = size

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

    def forward(self, x, zs):
        r"""Alias for :func:`model`"""
        # pylint: disable=arguments-differ
        return self.model(x, zs)
