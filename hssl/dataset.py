import numpy as np

import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes

import torch as t
import torch.utils.data


class Dataset(t.utils.data.Dataset):
    def __init__(
            self,
            image: t.Tensor,
            label: np.ndarray,
            data: pd.DataFrame,
            patch_size: int = 700,
    ):
        self.image = image
        self.label = label

        self.data = t.tensor(data.values).float()
        self.data = t.cat([t.zeros(self.data.shape[0])[:, None], self.data], 1)
        self.data = t.cat([t.zeros(self.data.shape[1])[None, :], self.data], 0)
        self.data[0, 0] = 1.

        self.h, self.w = [min(s, patch_size) for s in image.shape[-2:]]

    def __len__(self):
        return int(np.ceil(
            np.product(self.image.shape[-2:]) / self.h / self.w))

    def __getitem__(self, idx):
        y, x = [
            np.random.randint(s - d + 1)
            for s, d in zip(self.image.shape[-2:], (self.h, self.w))
        ]

        image = self.image[:, y:y + self.h, x:x + self.w].clone()
        label = self.label[y:y + self.h, x:x + self.w].copy()

        # remove partially visible labels
        label[np.invert(binary_fill_holes(label == 0))] = 0

        labels = [*sorted(np.unique(label))]
        data = self.data[labels, :]
        label = t.tensor(np.searchsorted(labels, label))

        return dict(
            image=image,
            label=label,
            data=data,
        )
