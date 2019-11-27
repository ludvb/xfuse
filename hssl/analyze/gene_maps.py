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


def compute_gene_maps(
    gene_name_regex: str = r".*", normalize: bool = True
) -> None:
    r"""Gene maps analysis function"""
    # pylint: disable=too-many-locals
    model = require("model")
    dataloader = require("dataloader")
    save_path = require("save_path")

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

    def _compute_gene_map_st(
        guide_trace, model_trace, data,  # pylint: disable=unused-argument
    ):
        rate_im = model_trace.nodes["rim"]["value"]
        if not normalize:
            rate_im *= model_trace.nodes["scale"]["value"]
        rate_mg = model_trace.nodes["rate_mg"]["value"]
        for name, rate_m in tqdm(
            zip(dataloader.dataset.genes, rate_mg.t()),
            total=len(dataloader.dataset.genes),
        ):
            gene_map = torch.einsum("fyx,f->yx", rate_im[0], rate_m.exp())
            gene_map -= gene_map.min()
            gene_map /= gene_map.max() - gene_map.min()
            gene_map = (255 * gene_map).round().byte()
            nonzero_labels = (data["data"][0].sum(1) == 0).nonzero() + 1
            gene_map[np.isin(data["label"].cpu(), nonzero_labels.cpu())] = 0
            yield name, gene_map

    fns: Dict[
        str,
        Callable[
            [pyro.poutine.Trace, pyro.poutine.Trace, Dict[str, Any]],
            Iterator[Tuple[str, torch.Tensor]],
        ],
    ] = {
        "ST": _compute_gene_map_st,
    }

    with Session(
        default_device=torch.device("cpu"), pyro_stack=[]
    ), torch.no_grad():
        for x in tqdm(dataloader, position=1):
            experiment_type = next(iter(x.keys()))
            slide_name = x[experiment_type]["slide_name"][0]

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

            for gene_name, gene_map in fns[experiment_type](
                guide_trace.trace, model_trace.trace, x[experiment_type]
            ):
                if not re.match(
                    gene_name_regex, gene_name, flags=re.IGNORECASE
                ):
                    continue
                filename = os.path.join(
                    save_path, os.path.basename(slide_name), f"{gene_name}.png"
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                imwrite(filename, gene_map.cpu().numpy())


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
