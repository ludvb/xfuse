import os
import re
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import pyro
import torch
from imageio import imwrite
from tqdm import tqdm

from .analyze import Analysis, _register_analysis
from ..data import Data, Dataset
from ..data.slide import FullSlide, Slide
from ..data.utility.misc import make_dataloader
from ..logging import WARNING, log
from ..session import Session, require
from ..utility import cleanup_mask
from ..utility.visualization import (
    greyscale2colormap,
    mask_background,
    balance_colors,
)


def compute_gene_maps(
    gene_name_regex: str = r".*", normalize: bool = False
) -> None:
    r"""Gene maps analysis function"""
    # pylint: disable=too-many-locals
    genes = require("genes")
    model = require("model")
    dataloader = require("dataloader")
    save_path = require("save_path")

    output_dir = os.path.join(save_path, "gene_maps")

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(
                        data=v.data,
                        # pylint: disable=unnecessary-lambda
                        # ^ Necessary for type checking to pass
                        iterator=lambda x: FullSlide(x),
                    )
                    for k, v in dataloader.dataset.data.slides.items()
                },
                design=dataloader.dataset.data.design,
            )
        ),
        batch_size=1,
        shuffle=False,
    )

    all_genes = np.array(genes)
    selected_genes = np.array(
        [
            x
            for x in all_genes
            if re.match(gene_name_regex, x, flags=re.IGNORECASE)
        ]
    )

    def _compute_gene_map_st(
        guide_trace, model_trace, data  # pylint: disable=unused-argument
    ):
        rate_im = model_trace.nodes["rim"]["value"]
        if not normalize:
            rate_im *= model_trace.nodes["scale"]["value"]
        rate_mg = model_trace.nodes["rate_mg"]["value"]
        rate_mg = rate_mg[:, np.isin(all_genes, selected_genes).nonzero()[0]]
        progress = tqdm(
            zip(selected_genes, rate_mg.t()), total=len(selected_genes)
        )
        mask = (
            model_trace.nodes["scale"]["value"]
            > 0.01 * model_trace.nodes["scale"]["value"].max()
        )
        mask = mask.squeeze().cpu().numpy()
        mask = cleanup_mask(mask, 0.01)
        for name, rate_m in progress:
            progress.set_description(name)
            gene_map = torch.einsum("fyx,f->yx", rate_im[0], rate_m.exp())
            yield name, gene_map, mask

    fns: Dict[
        str,
        Callable[
            [pyro.poutine.Trace, pyro.poutine.Trace, Dict[str, Any]],
            Iterator[Tuple[str, torch.Tensor, torch.Tensor]],
        ],
    ] = {"ST": _compute_gene_map_st}

    with Session(
        default_device=torch.device("cpu"), pyro_stack=[]
    ), torch.no_grad():
        progress = tqdm(dataloader, position=1)
        for x in progress:
            experiment_type = next(iter(x.keys()))
            slide_name = x[experiment_type]["slide_name"][0]
            progress.set_description(slide_name)

            if experiment_type not in fns.keys():
                log(
                    WARNING,
                    "Gene map analysis is not implemented for experiment type"
                    ' "%s".'
                    ' Sample "%s" will be skipped in this analysis.',
                    experiment_type,
                    slide_name,
                )
                continue

            with pyro.poutine.trace() as guide_trace:
                model.guide(x)
            with pyro.poutine.replay(trace=guide_trace.trace):
                with pyro.poutine.trace() as model_trace:
                    model.model(x)

            for gene_name, gene_map, mask in fns[experiment_type](
                guide_trace.trace, model_trace.trace, x[experiment_type]
            ):
                gene_map = balance_colors(
                    gene_map.cpu().numpy(), q=0, q_high=0.999
                )
                gene_map = greyscale2colormap(gene_map)
                gene_map = mask_background(gene_map, mask)
                filename = os.path.join(
                    output_dir,
                    os.path.basename(slide_name),
                    f"{gene_name}.png",
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                imwrite(filename, gene_map)


_register_analysis(
    name="gene_maps",
    analysis=Analysis(
        description=(
            "Constructs a map of imputed expression for each gene in the"
            " dataset."
        ),
        function=compute_gene_maps,
    ),
)
