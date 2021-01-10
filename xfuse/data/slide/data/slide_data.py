from __future__ import annotations
from abc import ABCMeta, abstractproperty
from typing import List


class SlideData(metaclass=ABCMeta):
    r"""Abstract class for different kinds of slide data"""

    @abstractproperty
    def data_type(self) -> str:
        r"""The type tag of this slide"""

    @abstractproperty
    def genes(self) -> List[str]:
        r"""Genes returned from this dataset"""

    @genes.setter
    def genes(self, genes: List[str]) -> SlideData:
        r"""Setter for which genes to return from this dataset"""
