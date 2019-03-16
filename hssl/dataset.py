import itertools as it

from typing import List

import numpy as np

import pandas as pd

from pyvips import Image

from scipy.ndimage.morphology import binary_fill_holes

import torch as t
import torch.utils.data

from torchvision import transforms
from torchvision.transforms.functional import affine, to_pil_image

from .image import to_array
from .utility import center_crop


class Slide(t.utils.data.Dataset):
    def __init__(
            self,
            image: Image,
            label: Image,
            data: pd.DataFrame,
            patch_size: int = 512,
    ):
        self.image = image
        self.label = label
        self.data = t.tensor(data.values).float()

        self.h, self.w = self.image.height, self.image.width

        assert(self.h == self.label.height and self.w == self.label.width)

        self.patch = patch_size
        if any(d < self._xpatch for d in (self.w, self.h)):
            raise ValueError(
                'image is too small for patch size '
                f'(needs to be at least {self._xpatch} px in each dimension)'
            )

        self.image_augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05,
            ),
        ])

    @property
    def _xpatch(self):
        return int(np.ceil(
            self.patch
            * np.sqrt(2)
        ))

    def __len__(self):
        return int(np.ceil(
            self.w * self.h / (self.patch ** 2)))

    def __getitem__(self, idx):
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

        # remove partially visible labels
        label[np.invert(binary_fill_holes(label == 0))] = 0

        labels = [*sorted(np.unique(label))]
        data = self.data[[x - 1 for x in labels if x > 0], :]
        label = np.searchsorted(labels, label)

        return dict(
            image=t.tensor(image / 255 * 2 - 1).permute(2, 0, 1).float(),
            label=t.tensor(label).long(),
            data=data,
        )


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
    ns, data = zip(*[(len(d), d) for d in (x.pop('data') for x in xs)])
    effects = [x.pop('effects').expand(n, -1) for n, x in zip(ns, xs)]
    nlabels = [len(d) for d in data]
    labels = [
        (l + n) * (l != 0).long() for l, n in
        zip((x.pop('label') for x in xs), it.accumulate((0, *nlabels)))
    ]
    return dict(
        data=t.cat(data),
        effects=t.cat(effects),
        label=t.stack(labels),
        **{k: t.stack([x[k] for x in xs]) for k in xs[0].keys()},
    )


def spot_size(dataset: Dataset):
    return np.mean(np.concatenate([
        np.bincount(d['label'].flatten())[1:]
        for d in dataset
    ]))
