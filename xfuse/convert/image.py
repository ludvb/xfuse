from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..utility.core import rescale
from ..utility.mask import compute_tissue_mask
from .utility import find_margin, write_data


def run(
    image: np.ndarray,
    output_file: str,
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
    mask: bool = True,
    custom_mask: Optional[np.ndarray] = None,
    rotate: bool = False,
) -> None:
    r"""
    Converts image data into the data format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    if scale_factor is not None:
        image = rescale(image, scale_factor, Image.BOX)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }
        if custom_mask is not None:
            custom_mask = rescale(custom_mask, scale_factor, Image.NEAREST)

    col_mask, row_mask = find_margin(image)
    image = image[row_mask][:, col_mask]
    if custom_mask is not None:
        custom_mask = custom_mask[row_mask][:, col_mask]

    if scale_factor is not None:
        # The outermost pixels may belong in part to the margin if we
        # downscaled the image. Therefore, remove one extra row/column.
        image = image[1:-1, 1:-1]
        if custom_mask is not None:
            custom_mask = custom_mask[1:-1, 1:-1]

    if mask:
        tissue_mask = compute_tissue_mask(image, initial_mask=custom_mask)
        label = np.array(tissue_mask == 0, dtype=np.int16)
    else:
        label = np.zeros(image.shape[:2], dtype=np.int16)

    counts = pd.DataFrame(
        index=pd.Series(np.unique(label[label != 0]), name="n")
    )

    write_data(
        counts,
        image,
        label,
        type_label="ST",
        annotation={
            k: (v, {x: str(x) for x in np.unique(v)})
            for k, v in annotation.items()
        },
        auto_rotate=rotate,
        path=output_file,
    )
