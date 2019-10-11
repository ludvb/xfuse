from abc import ABC, abstractmethod

import torch as t
from torch.utils.data import Dataset

import numpy as np

from pyvips import Image

from scipy.ndimage.morphology import binary_fill_holes


class Slide(ABC, Dataset):
    def __init__(self, data: t.Tensor, image: Image, label: Image):
        self.image = image
        self.label = label
        self.data = data

        # FIXME: torch sparse tensors don't support indexing. this can be
        # removed once https://github.com/pytorch/pytorch/pull/24937 has been
        # merged
        if self.data.layout is not t.strided:
            self.data = self.data.to_dense()

        self.h, self.w = self.image.height, self.image.width

        assert self.h == self.label.height and self.w == self.label.width

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
        data = self.data[[x - 1 for x in labels if x > 0]]
        if data.shape[0] == 0:
            return self.__getitem__((idx + 1) % len(self))
        label = np.searchsorted(labels, label)

        return dict(
            image=t.tensor(image / 255 * 2 - 1).permute(2, 0, 1).float(),
            label=t.tensor(label).long(),
            data=data,
            type="ST",
        )
