import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import affine, to_pil_image

from .slide import Slide
from ..utility import to_array
from ...utility import center_crop


__all__ = ["RandomSlide"]


class RandomSlide(Slide):
    def __init__(self, *args, patch_size: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_augmentation = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                )
            ]
        )
        self.patch = patch_size
        if any(d < self._xpatch for d in (self.w, self.h)):
            raise ValueError(
                "image is too small for patch size "
                f"(needs to be at least {self._xpatch} px in each dimension)"
            )

    @property
    def _xpatch(self):
        return int(np.ceil(self.patch * np.sqrt(2)))

    def __len__(self):
        return int(np.ceil(self.w * self.h / (self.patch ** 2)))

    def _get_patch(self, idx):
        y, x = [
            np.random.randint(s - self._xpatch + 1) for s in (self.h, self.w)
        ]
        image = to_pil_image(
            to_array(self.image.extract_area(x, y, self._xpatch, self._xpatch))
        )
        label = to_pil_image(
            to_array(self.label.extract_area(x, y, self._xpatch, self._xpatch))
        )

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
