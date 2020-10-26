from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..utility import compute_tissue_mask
from .utility import rescale, write_data


def run(
    tissue_image: np.ndarray,
    output_file: str,
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
    mask: bool = True,
    rotate: bool = False,
) -> None:
    r"""
    Converts image data into the data format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    if scale_factor is not None:
        tissue_image = rescale(tissue_image, scale_factor, Image.BICUBIC)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }

    if mask:
        mask = compute_tissue_mask(tissue_image)
        label = np.array(mask == 0, dtype=np.int16)
    else:
        label = np.zeros(tissue_image.shape[:2], dtype=np.int16)

    counts = pd.DataFrame(
        index=pd.Series(np.unique(label[label != 0]), name="n")
    )

    write_data(
        counts,
        tissue_image,
        label,
        type_label="ST",
        annotation=annotation,
        auto_rotate=rotate,
        path=output_file,
    )
