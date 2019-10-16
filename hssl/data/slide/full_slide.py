import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from ..utility import to_array
from .slide import Slide

__all__ = ["FullSlide"]


class FullSlide(Slide):
    """A :class:`Slide` that yields the entire (uncropped) sample"""

    # pylint: disable=too-few-public-methods

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
        image = to_pil_image(to_array(self.image))
        label = to_pil_image(to_array(self.label))

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
