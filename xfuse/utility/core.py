from typing import Any, ContextManager, Protocol, Tuple, TypeVar, Union

import warnings


__all__ = [
    "center_crop",
    "temp_attr",
]


ArrayType = TypeVar("S", bound="ArrayLike")


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


def center_crop(x: ArrayLike, target_shape: Tuple[int, ...]) -> ArrayLike:
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
