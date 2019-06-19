from abc import abstractmethod

from functools import reduce

import itertools as it

from typing import List

import numpy as np

import pandas as pd

from pyvips import Image

from scipy.ndimage.morphology import binary_fill_holes

import torch as t
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torchvision import transforms
from torchvision.transforms.functional import affine, to_pil_image

from .image import to_array
from .utility import center_crop


class Slide(t.utils.data.Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            image: Image,
            label: Image,
    ):
        self.image = image
        self.label = label
        self.data = t.tensor(data.values).float()

        self.h, self.w = self.image.height, self.image.width

        assert(self.h == self.label.height and self.w == self.label.width)

    @abstractmethod
    def _get_patch(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        image, label = self._get_patch(idx)

        # remove partially visible labels
        label[np.invert(binary_fill_holes(label == 0))] = 0

        labels = [*sorted(np.unique(label))]
        data = self.data[[x - 1 for x in labels if x > 0], :]
        label = np.searchsorted(labels, label)

        return dict(
            image=t.tensor(image / 255 * 2 - 1).permute(2, 0, 1).float(),
            label=t.tensor(label).long(),
            data=data,
            type='ST',
        )


class RandomSlide(Slide):
    def __init__(
            self,
            *args,
            patch_size: int = 512,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05,
            ),
        ])
        self.patch = patch_size
        if any(d < self._xpatch for d in (self.w, self.h)):
            raise ValueError(
                'image is too small for patch size '
                f'(needs to be at least {self._xpatch} px in each dimension)'
            )

    @property
    def _xpatch(self):
        return int(np.ceil(
            self.patch
            * np.sqrt(2)
        ))

    def __len__(self):
        return int(np.ceil(
            self.w * self.h / (self.patch ** 2)))

    def _get_patch(self, idx):
        y, x = [
            np.random.randint(s - self._xpatch + 1) for s in (self.h, self.w)]
        image = to_pil_image(to_array(self.image.extract_area(
            x, y, self._xpatch, self._xpatch)))
        label = to_pil_image(to_array(self.label.extract_area(
            x, y, self._xpatch, self._xpatch)))

        rotation = np.random.uniform(-180, 180)
        shear = 0  # np.random.uniform(-10, 10)

        image = affine(image, rotation, (0, 0), 1, shear)
        label = affine(label, rotation, (0, 0), 1, shear)

        image = self.image_augmentation(image)

        image = np.array(image)
        label = np.array(label)

        image = center_crop(image, (self.patch, self.patch))
        label = center_crop(label, (self.patch, self.patch))

        if np.random.rand() < 0.5:
            image = image[::-1]
            label = label[::-1]

        return image, label


class Dataset(t.utils.data.Dataset):
    def __init__(
            self,
            slides: List[Slide],
            design: pd.DataFrame,
    ):
        self.design = design
        self.slides = slides

        self.observations = pd.DataFrame(dict(
            sample=np.repeat(range(len(slides)), [len(x) for x in slides]),
            idx=np.concatenate([range(len(x)) for x in slides]),
        ))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        slide = self.observations['sample'].iloc[idx]
        return dict(
            **self.slides[slide].__getitem__(
                self.observations['idx'].iloc[idx]),
            effects=t.tensor(self.design[slide].values),
        )


def collate(xs):
    def _remove_key(v):
        v.pop('type')
        return v

    def _sort_key(x):
        return x['type']

    def _collate(ys):
        # we can't collate the count data as a tensor since its dimension will
        # differ between samples. therefore, we return it as a list instead.
        data = [y.pop('data') for y in ys]
        return {
            'data': data,
            **default_collate(ys),
        }

    return {
        k: _collate([_remove_key(v) for v in vs])
        for k, vs in it.groupby(sorted(xs, key=_sort_key), key=_sort_key)
    }


def spot_size(dataset: Dataset):
    return np.median(np.concatenate([
        np.bincount(d['label'].flatten())
        for d in dataset
    ]))
