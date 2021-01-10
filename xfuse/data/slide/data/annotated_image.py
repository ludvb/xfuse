from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import torch

from .slide_data import SlideData
from .st_slide import STSlide


class AnnotatedImage(SlideData):
    """
    Data class for annotated images that lack associated expression data.
    """

    def __init__(
        self,
        image: torch.Tensor,
        annotation: torch.Tensor,
        name: str = "Annotation",
        label_names: Optional[Dict[int, str]] = None,
    ):
        if label_names is None:
            label_names = {x: str(x) for x in torch.unique(annotation)}
        self.image = image
        self.label = annotation
        self.name = name
        self.label_names = label_names

    @property
    def label_names(self) -> np.ndarray:
        """Names corresponding to the integer labels in :attr:`label`"""
        return self._label_names_array

    @label_names.setter
    def label_names(self, x: Dict[int, str]):
        self._label_names = x
        self._label_names_array = np.full(
            max(self._label_names.keys()) + 1, "", dtype="object"
        )
        self._label_names_array[list(self._label_names.keys())] = list(
            self._label_names.values()
        )

    @property
    def data_type(self) -> str:
        return "AnnotatedImage"

    @property
    def genes(self) -> List[str]:
        return []

    @genes.setter
    def genes(self, genes: List[str]) -> AnnotatedImage:
        pass

    @classmethod
    def from_st_slide(
        cls, st_slide: STSlide, annotation_name: Optional[str] = None
    ) -> AnnotatedImage:
        """Creates an :class:`AnnotatedImage` from an :class:`STSlide`"""
        if annotation_name is None:
            annotation = st_slide.label[()]
            annotation_name = "n"
            label_names = None
        else:
            annotation, label_names = st_slide.annotation(annotation_name)
        image = st_slide.image[()]
        return cls(
            torch.as_tensor(image.astype(np.float32)),
            torch.as_tensor(annotation.astype(np.int64)),
            name=annotation_name,
            label_names=label_names,
        )
