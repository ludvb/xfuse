from abc import ABCMeta, abstractmethod
from typing import Optional

import torch


__all__ = ["StatsWriter"]


class StatsWriter(metaclass=ABCMeta):
    r"""Abstract class for stats writers"""

    @abstractmethod
    def add_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Writes histogram data"""

    @abstractmethod
    def add_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Writes image data"""

    @abstractmethod
    def add_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Writes image grid data"""

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Writes scalar data"""
