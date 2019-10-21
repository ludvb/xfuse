import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .slide import Slide

__all__ = ["FullSlide"]


class FullSlide(Slide):
    r"""A :class:`Slide` that yields the entire (uncropped) sample"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_augmentation = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                )
            ]
        )

    def __len__(self):
        return 1

    def _get_patch(self, idx):
        image = to_pil_image(self.image.to_array())
        label = to_pil_image(self.label.to_array())

        image = self.image_augmentation(image)

        image = np.array(image)
        label = np.array(label)

        if np.random.rand() < 0.5:
            image = image[::-1]
            label = label[::-1]

        if np.random.rand() < 0.5:
            image = image[:, ::-1]
            label = label[:, ::-1]

        return image, label
