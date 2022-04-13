import itertools as it
import re
import warnings
from typing import Any, Callable, Dict, Iterable, Tuple, cast

import numpy as np
import torch
from imageio import imwrite
from scipy.ndimage.morphology import binary_fill_holes

from .analyze import Analysis, _register_analysis
from .prediction import predict
from ..data import Data, Dataset
from ..data.slide import AnnotatedImage, FullSlideIterator, Slide
from ..data.utility.misc import make_dataloader
from ..logging import Progressbar
from ..session import Session, require
from ..utility.core import chunks_of, resize
from ..utility.file import chdir
from ..utility.visualization import (
    balance_colors,
    greyscale2colormap,
    mask_background,
)
from ..utility.tensor import to_device


def generate_gene_maps(
    num_samples: int = 10,
    genes_per_batch: int = 10,
    predict_mean: bool = True,
    normalize: bool = False,
    scale: float = 1.0,
) -> Iterable[Tuple[str, str, np.ndarray]]:
    """Generates gene maps on the active dataset"""

    genes = require("genes")
    dataloader = require("dataloader")

    if scale <= 0 or scale > 1.0:
        raise ValueError("Argument `scale` must be in (0, 1]")

    def _compute_annotation(shape):
        scaled_shape = [x * scale for x in shape]
        ys, xs = [
            np.floor(np.linspace(0, scaled_x, x, endpoint=False)).astype(int)
            for scaled_x, x in zip(scaled_shape, shape)
        ]
        annotation = 1 + torch.as_tensor((1 + xs.max()) * ys[:, None] + xs)
        label_names = {
            1 + (1 + xs.max()) * y + x: (y, x)
            for y in np.unique(ys)
            for x in np.unique(xs)
        }
        return annotation, label_names

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(
                        data=AnnotatedImage(
                            torch.as_tensor(v.data.image[()]),
                            annotation=annotation,
                            name="coordinates",
                            label_names=label_names,
                        ),
                        iterator=FullSlideIterator,
                    )
                    for k, v in dataloader.dataset.data.slides.items()
                    for annotation, label_names in [
                        _compute_annotation(v.data.label.shape)
                    ]
                },
                design=dataloader.dataset.data.design,
            )
        ),
        batch_size=1,
        shuffle=False,
    )

    def _process_batch(samples):
        rows, cols = [x + 1 for x in samples[0]["rownames"][-1]]
        data = torch.stack([x["data"] for x in samples])
        data = data.reshape(data.shape[0], rows, cols, data.shape[-1])
        for gene, gene_data in zip(
            samples[0]["colnames"], data.permute(3, 0, 1, 2)
        ):
            yield samples[0]["section"], gene, cast(
                np.ndarray, gene_data.cpu().numpy()
            )

    with Progressbar(
        chunks_of(genes, genes_per_batch),
        total=int(np.ceil(len(genes) / genes_per_batch)),
        leave=False,
    ) as progress:
        for genes_batch in progress:
            with Session(dataloader=dataloader, genes=genes_batch):
                for _, samples in it.groupby(
                    sorted(
                        [
                            to_device(x, device=torch.device("cpu"))
                            for x in predict(
                                num_samples=num_samples,
                                genes_per_batch=len(genes_batch),
                                predict_mean=predict_mean,
                                normalize_scale=normalize,
                                normalize_size=True,
                            )
                        ],
                        key=lambda x: x["section"],
                    ),
                    key=lambda x: x["section"],
                ):
                    yield from _process_batch(list(samples))


def _run_gene_maps_analysis(
    gene_regex: str = r".*",
    num_samples: int = 10,
    genes_per_batch: int = 10,
    predict_mean: bool = True,
    normalize: bool = False,
    mask_tissue: bool = True,
    scale: float = 1.0,
    writer: str = "image",
    writer_args: Dict[str, Any] = None,
) -> None:
    r"""Gene maps analysis function"""

    genes = require("genes")
    slides = require("dataloader").dataset.data.slides

    if writer_args is None:
        writer_args = {}

    def _save_image(gene, samples, tissue_mask, fileformat="jpg"):
        def _prepare(x):
            x = balance_colors(x, q=0, q_high=0.999)
            x = greyscale2colormap(x)
            if tissue_mask is not None:
                x = mask_background(x, tissue_mask)
            return x

        imwrite(f"{gene}_mean.{fileformat}", _prepare(samples.mean(0)))
        imwrite(f"{gene}_stdv.{fileformat}", _prepare(samples.std(0)))

        lfc = np.log2(samples.transpose(1, 2, 0)) - np.log2(
            samples.mean((1, 2))
        )
        imwrite(
            f"{gene}_invcv+.{fileformat}",
            _prepare(lfc.mean(-1).clip(0) / lfc.std(-1)),
        )

    def _save_tensor(gene, samples, tissue_mask):
        if tissue_mask is not None:
            samples[:, ~tissue_mask] = 0.0
        torch.save(samples, f"{gene}.pt")

    writers: Dict[str, Callable[..., None]] = {
        "image": _save_image,
        "tensor": _save_tensor,
    }
    try:
        write = writers[writer]
    except KeyError as exc:
        raise ValueError(
            'Invalid data format "{}" (choose between: {})'.format(
                writer, ", ".join(f'"{x}"' for x in writers)
            )
        ) from exc

    tissue_masks = {}
    if mask_tissue:
        for slide_name in slides:
            try:
                zero_count_idxs = np.where(
                    np.array(slides[slide_name].data.counts.todense()).sum(1)
                    == 0.0
                )[0]
                tissue_masks[slide_name] = binary_fill_holes(
                    np.isin(
                        slides[slide_name].data.label,
                        1 + zero_count_idxs,
                        invert=True,
                    )
                )
            except AttributeError:
                warnings.warn(f'Failed to mask "{slide_name}"')

    with Session(
        genes=[
            x for x in genes if re.match(gene_regex, x, flags=re.IGNORECASE)
        ]
    ):
        for slide_name, gene, samples in generate_gene_maps(
            num_samples=num_samples,
            genes_per_batch=genes_per_batch,
            predict_mean=predict_mean,
            normalize=normalize,
            scale=scale,
        ):
            try:
                tissue_mask = tissue_masks[slide_name]
                tissue_mask = resize(tissue_mask, samples.shape[1:])
            except KeyError:
                tissue_mask = None
            with chdir(slide_name):
                write(gene, samples, tissue_mask, **writer_args)


_register_analysis(
    name="gene_maps",
    analysis=Analysis(
        description=(
            "Constructs a map of imputed expression for each gene in the"
            " dataset."
        ),
        function=_run_gene_maps_analysis,
    ),
)
