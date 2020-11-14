import os
from typing import Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from . import StatsWriter
from ....logging import DEBUG, log
from ....session import get, require


__all__ = ["TensorboardWriter"]


class TensorboardWriter(StatsWriter):
    r"""Tensorboard stats writer"""

    def __init__(self):
        self.__summary_writer = None

    @property
    def _summary_writer(self) -> SummaryWriter:
        save_path = require("save_path")
        log_dir = os.path.join(save_path, "stats", "tb")
        if (
            self.__summary_writer is None
            or self.__summary_writer.log_dir != log_dir
        ):
            log(DEBUG, "Creating new SummaryWriter (log_dir = %s)", log_dir)
            self.__summary_writer = SummaryWriter(log_dir=log_dir)
        return self.__summary_writer

    def add_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Logs a histogram"""
        self._summary_writer.add_histogram(
            tag, values, global_step=get("training_data").step
        )

    def add_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image"""
        self._summary_writer.add_image(
            tag,
            img_tensor,
            global_step=get("training_data").step,
            dataformats="HWC",
        )

    def add_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image grid"""
        self._summary_writer.add_images(
            tag,
            img_tensor,
            global_step=get("training_data").step,
            dataformats="NHWC",
        )

    def add_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Logs a scalar"""
        self._summary_writer.add_scalar(
            tag, scalar_value, global_step=get("training_data").step
        )
