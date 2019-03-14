import numpy as np

from pyvips import Image


__all__ = ['to_array']


_vips2nptype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def to_array(image: Image) -> np.ndarray:
    return np.ndarray(
        buffer=image.write_to_memory(),
        shape=(image.height, image.width, image.bands),
        dtype=(
            {
                'uchar': np.uint8,
                'char': np.int8,
                'ushort': np.uint16,
                'short': np.int16,
                'uint': np.uint32,
                'int': np.int32,
                'float': np.float32,
                'double': np.float64,
                'complex': np.complex64,
                'dpcomplex': np.complex128,
            }
            [image.format]
        ),
    )
