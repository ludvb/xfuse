import itertools as it
from typing import (
    Any,
    ContextManager,
    Iterable,
    List,
    Protocol,
    Tuple,
    TypeVar,
    Sequence,
    Union,
)

import warnings

import numpy as np
from PIL import Image


__all__ = [
    "center_crop",
    "chunks_of",
    "rescale",
    "resize",
    "temp_attr",
]


ArrayType = TypeVar("ArrayType", bound="ArrayLike")


class ArrayLike(Protocol):
    r"""
    A protocol for sliceable objects (e.g., numpy arrays or pytorch tensors)
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        # pylint: disable=missing-docstring
        ...

    def __getitem__(
        self: ArrayType, idx: Union[slice, Tuple[slice, ...]]
    ) -> ArrayType:
        ...

    def __setitem__(
        self: ArrayType, idx: Union[slice, Tuple[slice, ...]], value: Any
    ) -> None:
        ...


def center_crop(x: ArrayType, target_shape: Tuple[int, ...]) -> ArrayType:
    r"""Crops `x` to the given `target_shape` from the center"""
    return x[
        tuple(
            [
                slice(round((a - b) / 2), round((a - b) / 2) + b)
                if b is not None
                else slice(None)
                for a, b in zip(x.shape, target_shape)
            ]
        )
    ]


def rescale(
    image: np.ndarray, scaling_factor: float, resample: int = Image.NEAREST
) -> np.ndarray:
    r"""
    Rescales image by a given `scaling_factor`

    :param image: Image array
    :param scaling_factor: Scaling factor
    :param resample: Resampling filter
    :returns: The rescaled image
    """
    image = Image.fromarray(image)
    image = image.resize(
        [round(x * scaling_factor) for x in image.size], resample=resample,
    )
    image = np.array(image)
    return image


def resize(
    image: np.ndarray,
    target_shape: Sequence[int],
    resample: int = Image.NEAREST,
) -> np.ndarray:
    r"""
    Resizes image to a given `target_shape`

    :param image: Image array
    :param target_shape: Target shape
    :param resample: Resampling filter
    :returns: The rescaled image
    """
    image = Image.fromarray(image)
    image = image.resize(target_shape[::-1], resample=resample)
    image = np.array(image)
    return image


def temp_attr(obj: object, attr: str, value: Any) -> ContextManager:
    r"""
    Creates a context manager for setting transient object attributes.

    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(x=1)
    >>> with temp_attr(obj, 'x', 2):
    ...     print(obj.x)
    2
    >>> print(obj.x)
    1
    """

    class _TempAttr:
        def __init__(self):
            self.__original_value = None

        def __enter__(self):
            self.__original_value = getattr(obj, attr)
            setattr(obj, attr, value)

        def __exit__(self, *_):
            if getattr(obj, attr) == value:
                setattr(obj, attr, self.__original_value)
            else:
                warnings.warn(
                    f'Attribute "{attr}" changed while in context.'
                    " The new value will be kept.",
                )

    return _TempAttr()


T = TypeVar("T")


def chunks_of(xs: Iterable[T], size: int) -> Iterable[List[T]]:
    r"""
    Yields size `size` chunks of `xs`.

    >>> list(chunks_of([1, 2, 3, 4], 2))
    [[1, 2], [3, 4]]
    """

    class _StopMarker:
        pass

    for chunk in it.zip_longest(*[iter(xs)] * size, fillvalue=_StopMarker):
        yield list(filter(lambda x: x is not _StopMarker, chunk))
