import gzip
import os
import time
from multiprocessing import Pool
from typing import Dict, Optional
from warnings import warn

import torch
from imageio import imwrite

from . import StatsWriter
from ....session import get, require
from ....utility.file import first_unique_filename


__all__ = ["FileWriter"]


class FileWriter(StatsWriter):
    r"""Stats writer emitting .jpg and .csv.gz files"""

    def __init__(self, num_workers: int = 1):
        self._worker_pool = Pool(num_workers)
        self._file_cons: Dict[str, gzip.GzipFile] = {}

    def add_histogram(
        self, tag: str, values: torch.Tensor, bins: Optional[int] = None
    ) -> None:
        r"""Logs a histogram"""
        warn(
            RuntimeWarning(
                "Histogram logging is not yet supported for this writer"
            )
        )

    def add_image(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image"""
        save_path = require("save_path")
        *prefix, name = tag.split("/")
        filename = first_unique_filename(
            os.path.join(
                save_path,
                "stats",
                "image",
                *prefix,
                f"{name}-{int(time.time())}.jpg",
            )
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._worker_pool.apply_async(
            imwrite, (filename, img_tensor.cpu().numpy()),
        )

    def add_images(self, tag: str, img_tensor: torch.Tensor) -> None:
        r"""Logs an image grid"""
        warn(
            RuntimeWarning(
                "Image grid logging is not yet supported for this writer"
            )
        )

    def add_scalar(self, tag: str, scalar_value: float) -> None:
        r"""Logs a scalar"""
        save_path = require("save_path")
        training_data = get("training_data")
        *prefix, name = tag.split("/")
        filename = os.path.join(save_path, "stats", *prefix, f"{name}.csv.gz")
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
