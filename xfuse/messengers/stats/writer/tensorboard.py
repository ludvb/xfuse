from typing import Dict, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from . import StatsWriter
from ....logging import DEBUG, log
from ....session import get


__all__ = ["TensorboardWriter"]


class TensorboardWriter(StatsWriter):
    r"""Tensorboard stats writer"""

    def __init__(self):
        self.__summary_writer = None

    @property
    def _summary_writer(self) -> SummaryWriter:
        log_dir = get("work_dir").full_path
        if (
            self.__summary_writer is None
            or self.__summary_writer.log_dir != log_dir
        ):
            log(DEBUG, "Creating new SummaryWriter (log_dir = %s)", log_dir)
            self.__summary_writer = SummaryWriter(log_dir=log_dir)
        return self.__summary_writer

    def write_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Logs a histogram"""
        self._summary_writer.add_histogram(
            tag, values, global_step=get("training_data").step
        )

    def write_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image"""
        self._summary_writer.add_image(
            tag,
            img_tensor,
            global_step=get("training_data").step,
            dataformats="HWC",
        )

    def write_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image grid"""
        self._summary_writer.add_images(
            tag,
            img_tensor,
            global_step=get("training_data").step,
            dataformats="NHWC",
        )

    def write_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Logs a scalar"""
        self._summary_writer.add_scalar(
            tag, scalar_value, global_step=get("training_data").step
        )

    def write_scalars(self, tag: str, scalar_values: Dict[str, float]) -> None:
        r"""Logs a set of associated scalars"""
        self._summary_writer.add_scalars(
            tag, scalar_values, global_step=get("training_data").step
        )
