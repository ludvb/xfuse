from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image
from pycpd import RigidRegistration
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture

from ..logging import DEBUG, INFO, WARNING, log
from ..utility import rescale
from .utility import (
    Spot,
    crop_image,
    labels_from_spots,
    mask_tissue,
    write_data,
)


def _find_keypoints(
    image: np.ndarray, **detection_params: Any
) -> Tuple[np.ndarray, np.ndarray]:
    # pylint: disable=no-member
    # ^ pylint fails to identify cv.* members
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 254
    params.thresholdStep = 1
    for name, value in detection_params.items():
        setattr(params, name, value)
    detector = cv.SimpleBlobDetector_create(params)
    scaling_factor = 2000 / np.max(image.shape[:2])
    image = rescale(image, scaling_factor, resample=Image.BILINEAR)
    image = cv.blur(image, (7, 7))
    kps = detector.detect(image)
    pts = np.array(
        [[x.pt[1] / scaling_factor, x.pt[0] / scaling_factor] for x in kps]
    )
    radii = 0.5 * np.array([x.size / scaling_factor for x in kps])
    return pts, radii


def _to_homogeneous(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((*x.shape[:-1], 1))], 1)


def _from_homogeneous(x: np.ndarray) -> np.ndarray:
    return x[..., :-1]


def _translate_homogeneous(translation_vector: np.ndarray) -> np.ndarray:
    y, x = translation_vector
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [y, x, 1.0]])


def _scale_homogeneous(scale_factor: float) -> np.ndarray:
    return np.diag([scale_factor, scale_factor, 1.0])


def _rotate_homogeneous(rotation_matrix: np.ndarray) -> np.ndarray:
    y = np.eye(3)
    y[:2, :2] = rotation_matrix
    return y


def _register_point_clouds(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    def _log(iteration: int, error: float, **_) -> None:
        log(DEBUG, f"  {iteration:03d} : {error:.4e}")

    pre_scale = (dst.max() - dst.min()) / (src.max() - src.min())
    src = src * pre_scale
    pre_translation = np.median(dst, 0) - np.median(src, 0)
    src = src + pre_translation

    _, (scale, rotation, translation) = RigidRegistration(
        X=dst, Y=src, max_iterations=10000,
    ).register(_log)

    transform = np.eye(3)
    transform = transform @ _scale_homogeneous(pre_scale)
    transform = transform @ _translate_homogeneous(pre_translation)
    transform = transform @ _scale_homogeneous(scale)
    transform = transform @ _rotate_homogeneous(rotation)
    transform = transform @ _translate_homogeneous(translation)

    return transform


def _compute_spot_coordinates(
    tissue_image: np.ndarray,
    genepix_data: pd.DataFrame,
    barcode_list: pd.DataFrame,
    slide_image: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    pts_img, _ = _find_keypoints(tissue_image)

    log(DEBUG, "Registering tissue image to GenePix data")
    img2genepix = _register_point_clouds(
        pts_img, genepix_data[genepix_data.Name == "FRAME"][["Y", "X"]].values
    )
    genepix2img = np.linalg.inv(img2genepix)

    spot_data = genepix_data[genepix_data.Name != "FRAME"].copy()
    pts_genepix = spot_data[["Y", "X"]].values
    pts_genepix = _to_homogeneous(pts_genepix)
    pts_genepix = pts_genepix @ genepix2img
    pts_genepix = _from_homogeneous(pts_genepix)
    spot_data["y"] = pts_genepix[:, 0]
    spot_data["x"] = pts_genepix[:, 1]
    spot_data["r"] = (
        0.5
        * genepix_data["Dia."]
        * np.sqrt(np.abs(np.linalg.det(genepix2img)))
    )
    spot_data.drop(columns=["X", "Y", "Dia."])

    barcode_list["Row"] = (barcode_list.Row - 1) // 2 + 1
    barcode_list = barcode_list.reset_index().set_index(["Row", "Column"])

    spot_data = spot_data.set_index(["Row", "Column"]).join(barcode_list)

    if slide_image is not None:
        pts_slide, radii_slide = _find_keypoints(
            1 - slide_image / slide_image.max()
        )

        labels_slide = GaussianMixture(2).fit_predict(
            radii_slide.reshape(-1, 1)
        )
        frame_label = np.argmax(
            (radii_slide @ np.eye(2)[labels_slide])
            / np.eye(2)[labels_slide].sum(0)
        )
        pts_slide_frame = pts_slide[labels_slide == frame_label]

        log(DEBUG, "Registering slide image to GenePix data")
        slide2genepix = _register_point_clouds(
            pts_slide_frame,
            genepix_data[genepix_data.Name == "FRAME"][["Y", "X"]].values,
        )
        slide2img = slide2genepix @ genepix2img

        pts_slide = _to_homogeneous(pts_slide)
        pts_slide = pts_slide @ slide2img
        pts_slide = _from_homogeneous(pts_slide)
        radii_slide = radii_slide * np.sqrt(np.abs(np.linalg.det(slide2img)))

        min_pdist = np.min(pdist(spot_data[["y", "x"]].values))
        pts_genepix_expanded = np.pad(
            spot_data[["y", "x"]].values, ((0, 0), (0, 1))
        )
        pts_slide_expanded = np.concatenate(
            [
                np.pad(pts_slide, ((0, 0), (0, 1))),
                np.pad(
                    spot_data[["y", "x"]].values,
                    ((0, 0), (0, 1)),
                    constant_values=min_pdist,
                ),
            ]
        )
        distances = cdist(pts_genepix_expanded, pts_slide_expanded)
        genepix_idxs, slide_idxs = linear_sum_assignment(distances)

        detected_mask = slide_idxs < pts_slide.shape[0]
        detected_slide_idxs = slide_idxs[detected_mask]
        detected_genepix_idxs = genepix_idxs[detected_mask]
        detected = pd.DataFrame(
            np.concatenate(
                [
                    pts_slide[detected_slide_idxs],
                    radii_slide[detected_slide_idxs].reshape(-1, 1),
                ],
                axis=1,
            ),
            columns=["y_slide", "x_slide", "r_slide"],
            index=spot_data.index[detected_genepix_idxs],
        )

        spot_data = spot_data.join(detected)

        replacements = spot_data.Flags & ~np.isnan(spot_data.r_slide)

        log(INFO, "Replacements:")
        for _, values in spot_data[replacements][
            ["x", "y", "r", "x_slide", "y_slide", "r_slide"]
        ].iterrows():
            log(
                INFO,
                "  (X=%.1f, Y=%.1f, D=%.1f) → (%.1f, %.1f, %.1f)",
                *values,
            )

        spot_data.loc[replacements, "x"] = spot_data.x_slide[replacements]
        spot_data.loc[replacements, "y"] = spot_data.y_slide[replacements]
        spot_data.loc[replacements, "r"] = spot_data.r_slide[replacements]
        spot_data.loc[replacements, "Flags"] = 0

    unrecoverable = spot_data[spot_data.Flags != 0]
    if unrecoverable.shape[0] > 0:
        log(
            WARNING,
            "The following barcodes were unrecoverable: %s",
            ", ".join(unrecoverable.Barcode),
        )
    spot_data = spot_data.drop(unrecoverable.index)

    return spot_data[["Barcode", "x", "y", "r"]].set_index("Barcode")


def run(
    tissue_image: np.ndarray,
    bc_matrix: h5py.File,
    genepix_data: pd.DataFrame,
    barcode_list: pd.DataFrame,
    output_file: str,
    slide_image: Optional[np.ndarray],
    annotation: Optional[Dict[str, np.ndarray]] = None,
    scale_factor: Optional[float] = None,
    mask: bool = True,
) -> None:
    r"""
    Converts data from the 10X SpaceRanger pipeline for visium arrays into
    the data format used by xfuse.
    """
    if annotation is None:
        annotation = {}

    counts = csr_matrix(
        (
            bc_matrix["matrix"]["data"],
            bc_matrix["matrix"]["indices"],
            bc_matrix["matrix"]["indptr"],
        ),
        shape=(
            bc_matrix["matrix"]["barcodes"].shape[0],
            bc_matrix["matrix"]["features"]["name"].shape[0],
        ),
    )
    counts = pd.DataFrame(
        counts.todense().astype(float),
        columns=bc_matrix["matrix"]["features"]["name"][()].astype(str),
        index=pd.Index(
            [
                x.decode().split("-")[0]
                for x in bc_matrix["matrix"]["barcodes"][()]
            ],
            name="Barcode",
        ),
    )

    if scale_factor is not None:
        tissue_image = rescale(tissue_image, scale_factor, Image.BICUBIC)
        annotation = {
            k: rescale(v, scale_factor, Image.NEAREST)
            for k, v in annotation.items()
        }

    spot_coordinates = _compute_spot_coordinates(
        tissue_image, genepix_data, barcode_list, slide_image
    )
    barcodes = np.intersect1d(counts.index, spot_coordinates.index)
    counts = counts.loc[barcodes]
    spot_coordinates = spot_coordinates.loc[barcodes]
    counts = counts.set_index(
        pd.Index(np.arange(counts.shape[0], dtype=np.uint16) + 1, name="n")
    )
    spots = spot_coordinates.apply(
        lambda x: Spot(x=x.x, y=x.y, r=x.r), axis=1
    ).tolist()
    label = np.zeros(tissue_image.shape[:2]).astype(np.int16)
    labels_from_spots(label, spots)

    tissue_image = crop_image(tissue_image, spots)
    label = crop_image(label, spots)
    annotation = {k: crop_image(v, spots) for k, v in annotation.items()}

    if mask:
        counts, label = mask_tissue(tissue_image, counts, label)

    write_data(
        counts,
        tissue_image,
        label,
        type_label="ST",
        annotation=annotation,
        path=output_file,
    )
