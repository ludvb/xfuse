from __future__ import annotations
from typing import List

import numpy as np
import torch

from ....session import Session, get
from ....session.items.rng_state import RNGState
from ...synthetic import Circle, Square, Triangle, generate_synthetic_data
from .slide_data import SlideData


class SyntheticSlide(SlideData):
    def __init__(
        self,
        num_molecules: int = 32,
        molecule_size: int = 16,
        num_tiles: int = 4,
        tile_size: int = 32,
        reads_per_pixel: float = 10.0,
        reads_per_bg_pixel: float = 1.0,
        monochrome: bool = False,
        poisson_noise: bool = False,
        image_noise: float = 0.0,
        concentration: float = np.finfo(np.float32).max,
        knockout: Optional[List[str]] = None,
        discard_expression_data: bool = False,
        allow_edge_clipping: bool = False,
    ):
        self.num_molecules = num_molecules
        self.molecule_size = molecule_size
        self.num_tiles = num_tiles
        self.tile_size = tile_size
        self.reads_per_pixel = reads_per_pixel
        self.reads_per_bg_pixel = reads_per_bg_pixel
        self.monochrome = monochrome
        self.poisson_noise = poisson_noise
        self.image_noise = image_noise
        self.concentration = concentration
        self.knockout = knockout or []
        self.discard_expression_data = discard_expression_data
        self.allow_edge_clipping = allow_edge_clipping
        self.reset_data()

    @property
    def data_type(self) -> str:
        return "ST"

    @property
    def genes(self):
        return list(self.__counts.columns.values)

    @genes.setter
    def genes(self, genes: List[str]) -> SyntheticSlide:
        if genes != self.genes:
            raise NotImplementedError()
        return self

    def reset_data(self) -> SyntheticSlide:
        if get("eval"):
            rng_state = RNGState(hash(id(self)))
        else:
            rng_state = get("rng_state")
        with Session(rng_state=rng_state):
            (
                self.__image,
                self.__label,
                self.__counts_ground_truth,
                self.__counts,
            ) = generate_synthetic_data(
                self.num_molecules,
                self.molecule_size,
                [
                    Circle(
                        np.array([1.0, 1.0, 1.0])
                        if self.monochrome
                        else np.array([1.0, 0.0, 0.0]),
                    ),
                    Square(
                        np.array([1.0, 1.0, 1.0])
                        if self.monochrome
                        else np.array([0.0, 1.0, 0.0]),
                    ),
                    Triangle(
                        np.array([1.0, 1.0, 1.0])
                        if self.monochrome
                        else np.array([0.0, 0.0, 1.0]),
                    ),
                ],
                self.num_tiles,
                self.tile_size,
                reads_per_pixel=self.reads_per_pixel,
                reads_per_bg_pixel=self.reads_per_bg_pixel,
                poisson_noise=self.poisson_noise,
                image_noise_level=self.image_noise,
                concentration=self.concentration,
                allow_edge_clipping=self.allow_edge_clipping,
                knockout=self.knockout,
            )
        if self.discard_expression_data:
            self.__label[...] = 0
            self.__counts = self.__counts.drop(self.__counts.index)
        self.__image = 0.7 * self.__image
        return self

    @property
    def counts(self):
        r"""Getter for the count data"""
        return self.__counts.values

    @property
    def image(self):
        r"""Getter for the slide image"""
        return self.__image

    @property
    def label(self):
        r"""Getter for the label image of the slide"""
        return self.__label

    def annotation(self, name):
        raise NotImplementedError()

    def extra(self, name):
        if name == "ground_truth":
            return self.__counts_ground_truth
        if name == "ground_truth_genes":
            return self.__counts.columns.values.astype("S")
        raise RuntimeError(f'Extra layer "{name}" is missing')

    def prepare_data(self, image, label):
        labels = np.unique(label[label != 0])
        data = self.counts[(labels - 1).tolist()]
        label = np.searchsorted([0, *labels], label)

        return dict(
            image=torch.as_tensor(image).float(),
            label=torch.as_tensor(label).long(),
            data=torch.as_tensor(data).float(),
        )
