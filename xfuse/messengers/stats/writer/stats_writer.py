from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import torch


__all__ = ["StatsWriter"]


class StatsWriter(metaclass=ABCMeta):
    r"""Abstract class for stats writers"""

    @abstractmethod
    def write_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Writes histogram data"""

    @abstractmethod
    def write_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Writes image data"""

    @abstractmethod
    def write_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Writes image grid data"""

    @abstractmethod
    def write_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Writes scalar data"""

    @abstractmethod
    def write_scalars(self, tag: str, scalar_values: Dict[str, float]) -> None:
        r"""Writes scalar data"""
