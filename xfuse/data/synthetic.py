import itertools as it
import warnings
from abc import ABCMeta, abstractproperty
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from skimage.transform import rotate

from ..logging import WARNING, log
from ..utility.tensor import sparseonehot


class Molecule(metaclass=ABCMeta):
    @abstractproperty
    def shape(self) -> np.ndarray:
        pass


class Circle(Molecule):
    def __init__(self, color: np.ndarray, num_points=360):
        self.color = color
        angles = np.linspace(0, 2 * np.pi, num=num_points, endpoint=False)
        self.__points = np.stack([np.sin(angles), np.cos(angles)], -1)

    @property
    def shape(self):
        return self.__points


class Square(Molecule):
    def __init__(self, color: np.ndarray):
        self.color = color
        self.__points = 2 ** -0.5 * np.array(
            [[-1.0, -1.0], [-1.0, +1.0], [+1.0, +1.0], [+1.0, -1.0]]
        )

    @property
    def shape(self):
        return self.__points


class Triangle(Molecule):
    def __init__(self, color: np.ndarray):
        self.color = color
        self.__points = np.array(
            [[-0.5, -np.sqrt(3 / 4)], [-0.5, +np.sqrt(3 / 4)], [+1.0, +0.0]]
        )

    @property
    def shape(self):
        return self.__points


def rotate_shape(x: np.ndarray, a: float) -> np.ndarray:
    r_mat = np.array([[+np.cos(a), +np.sin(a)], [-np.sin(a), +np.cos(a)]])
    return x @ r_mat


def _make_grid(stacked_tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return (
        stacked_tiles.reshape(
            rows,
            cols,
            stacked_tiles.shape[1],
            stacked_tiles.shape[2],
            stacked_tiles.shape[3],
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(
            rows * stacked_tiles.shape[1],
            rows * stacked_tiles.shape[2],
            stacked_tiles.shape[3],
        )
    )


def _embed_shape(shape, width: int, height: int, x: int, y: int) -> np.ndarray:
    embedded_shape = np.zeros((height, width), dtype=bool)
    embedded_shape[
        (y - shape.shape[0] // 2) : (y + (shape.shape[0] + 1) // 2),
        x - shape.shape[1] // 2 : x + (shape.shape[1] + 1) // 2,
    ] = shape
    return embedded_shape


def generate_synthetic_data(
    num_molecules: int,
    molecule_size: int,
    molecules: List[Molecule],
    num_tiles: int,
    tile_size: int,
    reads_per_pixel: float = 10.0,
    reads_per_bg_pixel: float = 1.0,
    poisson_noise: bool = False,
    image_noise_level: float = 0.0,
    concentration: float = 1.0,
    allow_overlaps: bool = False,
    allow_edge_clipping: bool = False,
    max_attempts: int = 100,
    knockout: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label = (
        np.ones((num_tiles * num_tiles, tile_size, tile_size), dtype=np.uint16)
        * (1 + np.arange(num_tiles * num_tiles))[:, None, None]
    )

    label = _make_grid(label[..., None], num_tiles, num_tiles).squeeze(-1)
    label = np.pad(label, 1)

    size = label.shape

    image = np.zeros((*size, 3), dtype=np.float32)
    counts = np.zeros((*size, len(molecules)), dtype=np.float32)

    probs = np.random.dirichlet(np.ones(len(molecules)) * concentration)
    samples = np.random.choice(
        len(molecules), num_molecules, replace=True, p=probs
    )
    for idx in samples:
        molecule = molecules[idx]
        margin = 0.0 if allow_edge_clipping else (molecule_size + 1) / 2
        for attempt in it.count(1):
            x, y = (
                np.random.uniform(size=2)
                * (np.array(image.shape[:2]) - 2 * margin)
                + margin
            )
            angle = np.random.uniform(high=2 * np.pi)
            pts = rotate_shape(molecule.shape, angle)
            pts = pts / 2 * molecule_size
            pts = pts + np.array([y, x])
            pts = pts.round().astype(np.int32)
            pts = pts.reshape(-1, 1, 2)
            shape_mask = cv.fillPoly(
                np.zeros((size[0], size[1]), dtype=np.uint8), [pts], 1,
            )
            shape_mask = shape_mask.astype(bool)
            if allow_overlaps or image[shape_mask].sum() == 0:
                image[shape_mask] += molecule.color
                counts[shape_mask] += (
                    np.eye(len(molecules))[idx] * reads_per_pixel
                )
                break
            if attempt >= max_attempts:
                log(
                    WARNING,
                    "Giving up placing molecule after %d attempts",
                    attempt,
                )
                break

    counts[counts == 0] = reads_per_bg_pixel

    count_matrix = (
        (
            sparseonehot(torch.as_tensor(label[label != 0]).long() - 1)
            .float()
            .t()
            @ torch.as_tensor(counts[label != 0])
        )
        .cpu()
        .numpy()
    )
    if poisson_noise:
        count_matrix = np.random.poisson(count_matrix)
    count_matrix = pd.DataFrame(
        count_matrix,
        columns=[f"mol{i:d}" for i, _ in enumerate(molecules, 1)],
        index=1 + np.arange(num_tiles * num_tiles),
    )

    for c in knockout:
        try:
            gene_channel, *_ = (count_matrix.columns.values == c).nonzero()[0]
            assert _ == []
            counts[..., gene_channel] = 0
            count_matrix[c][:] = 0
        except ValueError:
            print(
                f'Invalid gene: {c} (choose between: {", ".join(count_matrix.columns.values)})'
            )

    image = 2 * image - 1
    image = image.clip(min=-1.0, max=1.0)
    image = 0.7 * image

    image_noise = np.random.normal(
        loc=0.0, scale=image_noise_level, size=image.shape
    ).astype(np.float32)
    image = image + image_noise

    return image, label, counts, count_matrix
