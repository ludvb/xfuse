import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import (
    _get_inverse_affine_matrix,
    affine,
    to_pil_image,
)

from ...utility import center_crop
from .slide import Slide

__all__ = ["RandomSlide"]


class RandomSlide(Slide):
    r"""
    A :class:`Slide` that yields randomly cropped patches of the sample
    """

    def __init__(
        self, *args, patch_size: int = 512, max_shear: float = 10, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_augmentation = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                )
            ]
        )
        self._max_shear = max_shear
        self._patch_x = self._patch_y = patch_size
        if any(a < b for a, b in zip((self.W, self.H), self._aug_patch)):
            raise ValueError(
                "image is too small for patch size"
                + " (needs to be at least "
                + "x".join(map(str, self._aug_patch))
                + " px)"
            )

    @property
    def _aug_patch(self):
        transform = np.concatenate(
            [
                np.array(
                    _get_inverse_affine_matrix(
                        (0.5, 0.5), 45.0, (0, 0), 1.0, self._max_shear
                    )
                ).reshape(2, -1),
                np.array([[0.0, 0.0, 1.0]]),
            ]
        )
        corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        inv_corners = transform @ np.transpose(corners)
        xmax, ymax = inv_corners[:2].max(1)
        xmin, ymin = inv_corners[:2].min(1)
        xaug = int(np.ceil((xmax - xmin) * self._patch_x))
        yaug = int(np.ceil((ymax - ymin) * self._patch_y))
        return xaug, yaug

    def __len__(self):
        xaug, yaug = self._aug_patch
        return int(np.ceil(self.W / xaug * self.H / yaug))

    def _get_patch(self, idx):
        xaug, yaug = self._aug_patch
        x, y = [
            np.random.randint(a - b + 1)
            for a, b in zip((self.W, self.H), (xaug, yaug))
        ]
        image = to_pil_image(self.image.extract(x, y, xaug, yaug).to_array())
        label = to_pil_image(self.label.extract(x, y, xaug, yaug).to_array())

        rotation = np.random.uniform(-180, 180)
        shear = np.random.uniform(-self._max_shear, self._max_shear)

        image = affine(image, rotation, (0, 0), 1, shear)
        label = affine(label, rotation, (0, 0), 1, shear)

        image = self.image_augmentation(image)

        image = np.array(image)
        label = np.array(label)

        image = center_crop(image, (self._patch_y, self._patch_x))
        label = center_crop(label, (self._patch_y, self._patch_x))

        if np.random.rand() < 0.5:
            image = image[::-1]
            label = label[::-1]

        return image, label
