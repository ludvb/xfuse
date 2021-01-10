from typing import List

import numpy as np
from skimage.util import pad

from ..data.synthetic import Circle, Square, Triangle, generate_synthetic_data
from .utility import write_data


def run(
    output_file: str,
    tile_size: int = 64,
    tile_padding: int = 16,
    num_tiles: int = 32,
    num_molecules: int = 2048,
    molecule_size: int = 16,
    reads_per_pixel: float = 10.0,
    reads_per_bg_pixel: float = 1.0,
    knockout: List[str] = [],
    poisson_noise: bool = False,
    allow_overlaps: bool = False,
    color_intensity: float = 0.5,
    discard_expression_data: bool = False,
    rotate: bool = False,
) -> None:
    molecules = [
        Circle(color_intensity * np.array([1.0, 0.0, 0.0])),
        Square(color_intensity * np.array([0.0, 1.0, 0.0])),
        Triangle(color_intensity * np.array([0.0, 0.0, 1.0])),
    ]
    image, label, counts, count_matrix = generate_synthetic_data(
        num_molecules,
        molecule_size,
        molecules,
        num_tiles,
        tile_size,
        reads_per_pixel,
        poisson_noise=poisson_noise,
        concentration=np.finfo(np.float32).max,
        knockout=knockout,
    )
    if discard_expression_data:
        label[...] = 0
        count_matrix = count_matrix.drop(count_matrix.index)
    write_data(
        count_matrix,
        image,
        label,
        type_label="ST",
        annotation={},
        extra={
            "ground_truth": (counts, True),
            "ground_truth_genes": (
                count_matrix.columns.values.astype("S"),
                False,
            ),
        },
        auto_rotate=rotate,
        crop=rotate,
        normalize_image=False,
        path=output_file,
    )
