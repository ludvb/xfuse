from abc import ABC, abstractmethod

import numpy as np
import torch
from pyvips import Image
from scipy.ndimage.morphology import binary_fill_holes
from torch.utils.data import Dataset


class Slide(ABC, Dataset):
    """
    Abstract class yielding observations from a single sample (tissue slide)
    """

    def __init__(self, data: torch.Tensor, image: Image, label: Image):
        self.image = image
        self.label = label
        self.data = data

        # FIXME: torch sparse tensors don't support indexing. this can be
        # removed once https://github.com/pytorch/pytorch/pull/24937 has been
        # merged
        if self.data.is_sparse:  # type: ignore
            self.data = self.data.to_dense()

        self.H, self.W = self.image.height, self.image.width

        assert self.H == self.label.height and self.W == self.label.width

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
            image=torch.tensor(image / 255 * 2 - 1).permute(2, 0, 1).float(),
            label=torch.tensor(label).long(),
            data=data,
            type="ST",
        )
