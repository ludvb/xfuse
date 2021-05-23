import gzip
import os
import time
from typing import Dict, Optional
from warnings import warn

import numpy as np
import torch
from imageio import imwrite

from . import StatsWriter
from ....session import get
from ....utility.file import first_unique_filename
from ....utility.visualization import _normalize


__all__ = ["FileWriter"]


class FileWriter(StatsWriter):
    r"""Stats writer emitting .jpg and .csv.gz files"""

    def __init__(self):
        self._file_cons: Dict[str, gzip.GzipFile] = {}

    def write_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Logs a histogram"""
        warn(
            RuntimeWarning(
                "Histogram logging is not yet supported for this writer"
            )
        )

    def write_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image"""
        training_data = get("training_data")
        *prefix, name = tag.split("/")
        filename = first_unique_filename(
            os.path.join(
                *prefix,
                f"{name}-{training_data.epoch}-{training_data.step}.png",
            )
        )
        if (dirname := os.path.dirname(filename)) != "":
            os.makedirs(dirname, exist_ok=True)
        img = img_tensor.detach().cpu().numpy()
        img = _normalize(img)
        img = (255 * img).astype(np.uint8)
        imwrite(os.path.abspath(filename), img)

    def write_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image grid"""
        N, H, W, C = img_tensor.shape  # pylint: disable=invalid-name
        cols = int(np.ceil(N ** 0.5))
        rows = int(np.ceil(N / cols))
        img_tensor = torch.cat(
            [img_tensor, torch.zeros(rows * cols - N, H, W, C).to(img_tensor)],
        )
        img_tensor = (
            img_tensor.reshape(rows, cols, H, W, C)
            .permute(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C)
        )
        self.write_image(tag, img_tensor)

    def write_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Logs a scalar"""
        training_data = get("training_data")
        *prefix, name = tag.split("/")
        filename = os.path.join(*prefix, f"{name}.csv.gz")
        if filename not in self._file_cons:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if not os.path.exists(filename):
                with gzip.open(filename, "wb") as fcon:
                    fcon.write("time,epoch,step,value\n".encode())
            self._file_cons[filename] = gzip.open(filename, "ab")
        self._file_cons[filename].write(
            str.encode(
                "{:f},{:d},{:d},{:f}\n".format(
                    time.time(),
                    training_data.epoch,
                    training_data.step,
                    scalar_value,
                )
            )
        )

    def write_scalars(self, tag: str, scalar_values: Dict[str, float]) -> None:
        r"""Logs a set of associated scalars"""
        training_data = get("training_data")
        *prefix, name = tag.split("/")
        filename = os.path.join(*prefix, f"{name}.csv.gz")
        if filename not in self._file_cons:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if not os.path.exists(filename):
                with gzip.open(filename, "wb") as fcon:
                    fcon.write("time,epoch,step,name,value\n".encode())
            self._file_cons[filename] = gzip.open(filename, "ab")
        for name, value in scalar_values.items():
            self._file_cons[filename].write(
                str.encode(
                    "{:f},{:d},{:d},{:s},{:f}\n".format(
                        time.time(),
                        training_data.epoch,
                        training_data.step,
                        name,
                        value,
                    )
                )
            )
