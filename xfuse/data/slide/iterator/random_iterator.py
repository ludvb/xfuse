from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import (
    _get_inverse_affine_matrix,
    to_pil_image,
)

from ....utility import center_crop
from ..data import STSlide
from ..iterator import SlideIterator


class RandomIterator(SlideIterator):
    r"""
    A :class:`SlideIterator` that yields randomly cropped patches of the sample
    """

    def __init__(
        self,
        slide: STSlide,
        patch_size: Optional[Tuple[float, float]] = None,
        max_rotation_jitter: float = 180.0,
        max_scale_jitter: float = 0.05,
        max_shear_jitter: float = 10.0,
    ):
        self._slide = slide
        self.image_augmentation = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                )
            ]
        )
        self._max_rotation_jitter = max_rotation_jitter
        self._max_scale_jitter = max_scale_jitter
        self._max_shear_jitter = max_shear_jitter
        if patch_size is None:
            patch_size = self._slide.W, self._slide.H
        self._patch_w, self._patch_h = patch_size

    @staticmethod
    def _compute_extended_patch_size(
        w: float, h: float, rotation: float, scale: float, shear: List[float]
    ) -> Tuple[float, float]:
        transform = np.concatenate(
            [
                np.array(
                    _get_inverse_affine_matrix(
                        center=(0.5, 0.5),
                        angle=rotation,
                        translate=(0, 0),
                        scale=scale,
                        shear=shear,
                    )
                ).reshape(2, -1),
                np.array([[0.0, 0.0, 1.0]]),
            ]
        )
        corners = np.array([[0, 0, 1], [0, h, 1], [w, 0, 1], [w, h, 1]])
        inv_corners = transform @ np.transpose(corners)
        xmax, ymax = inv_corners[:2].max(1)
        xmin, ymin = inv_corners[:2].min(1)
        return xmax - xmin, ymax - ymin

    def __len__(self):
        return int(
            np.ceil(
                self._slide.W / self._patch_w * self._slide.H / self._patch_h
            )
        )

    def __getitem__(self, idx):
        # pylint: disable=too-many-locals

        # Sample a random transformation
        rotation = np.random.uniform(
            -self._max_rotation_jitter, self._max_rotation_jitter
        )
        scale = np.exp(
            np.random.uniform(-self._max_scale_jitter, self._max_scale_jitter)
        )
        shear = np.random.uniform(
            -self._max_shear_jitter, self._max_shear_jitter, size=2
        )

        # Compute the "extended" patch size. This is the size of the patch that
        # we will first transform and then center crop to the final size.
        extpatch_w, extpatch_h = self._compute_extended_patch_size(
            w=self._patch_w,
            h=self._patch_h,
            rotation=rotation,
            scale=scale,
            shear=shear,
        )

        # The slide may not be large enough for the extended patch size. In
        # this case, we will downscale the target patch size until the extended
        # patch size fits.
        adjmul = min(
            1.0, self._slide.W / extpatch_w, self._slide.H / extpatch_h
        )
        extpatch_w = min(int(np.ceil(extpatch_w * adjmul)), self._slide.W)
        extpatch_h = min(int(np.ceil(extpatch_h * adjmul)), self._slide.H)
        patch_w = int(self._patch_w * adjmul)
        patch_h = int(self._patch_h * adjmul)

        # Extract the extended patch by sampling uniformly from the size of the
        # slide
        x, y = [
            np.random.randint(a - b + 1)
            for a, b in zip(
                (self._slide.W, self._slide.H), (extpatch_w, extpatch_h)
            )
        ]
        image = self._slide.image[y : y + extpatch_h, x : x + extpatch_w]
        image = (255 * (image + 1) / 2).astype(np.uint8)
        image = to_pil_image(image)
        label = to_pil_image(
            self._slide.label[y : y + extpatch_h, x : x + extpatch_w]
        )

        # Apply augmentations
        output_size = (max(extpatch_w, patch_w), max(extpatch_h, patch_h))
        transformation = _get_inverse_affine_matrix(
            center=(image.size[0] * 0.5, image.size[1] * 0.5),
            angle=rotation,
            translate=[(a - b) / 2 for a, b in zip(output_size, image.size)],
            scale=scale,
            shear=shear,
        )
        image = self.image_augmentation(image)
        image = np.array(
            image.transform(
                output_size,
                Image.AFFINE,
                transformation,
                resample=Image.BILINEAR,
            )
        )
        image = center_crop(image, (patch_h, patch_w))
        label = np.array(
            label.transform(
                output_size,
                Image.AFFINE,
                transformation,
                resample=Image.NEAREST,
            )
        )
        label = center_crop(label, (patch_h, patch_w))
        if np.random.rand() < 0.5:
            image = np.flip(image, 0).copy()
            label = np.flip(label, 0).copy()

        # Convert image to the correct data format (float32 in [-1, 1] and in
        # CHW order)
        image = 2 * image.astype(np.float32) / 255 - 1
        image = image.transpose(2, 0, 1)

        return self._slide.prepare_data(image, label)
