import os

import numpy as np
import pandas as pd
import pyro
import torch
from tqdm import tqdm

from ..data import Data, Dataset
from ..data.slide import FullSlide, Slide
from ..data.utility.misc import make_dataloader
from ..logging import WARNING, log
from ..session import Session, require
from .analyze import Analysis, _register_analysis


def _impute(
    slide: Slide,
    design: pd.Series,
    imputation_label: torch.Tensor,
    normalize_scale: bool = False,
    normalize_size: bool = False,
) -> torch.Tensor:
    model = require("model")

    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    design.name: Slide(
                        data=slide.data,
                        # pylint: disable=unnecessary-lambda
                        # ^ Necessary for type checking to pass
                        iterator=lambda x: FullSlide(x),
                    )
                },
                design=design.to_frame(),
            )
        )
    )

    data = next(iter(dataloader))
    data_type, *__this_should_be_empty = list(data.keys())
    assert __this_should_be_empty == []

    if data_type != "ST":
        raise NotImplementedError(
            f'Imputation not implemented for slide of type "{data_type}"'
        )

    with pyro.poutine.trace() as guide_trace:
        model.guide(data)

    data_ = data.copy()
    data_[data_type]["label"] = imputation_label.unsqueeze(0).to(
        data_[data_type]["label"]
    )
    data_[data_type]["data"] = [
        torch.zeros(
            int(imputation_label.max().item()),
            int(data_[data_type]["data"][0].shape[1]),
        ).to(data_[data_type]["data"][0])
    ]

    model_ = model.model
    if normalize_scale:
        model_ = pyro.poutine.condition(
            model_,
            # pylint: disable=not-callable
            {"scale": torch.tensor(1.0)},
        )

    with pyro.poutine.replay(trace=guide_trace.trace):
        with pyro.poutine.trace() as model_trace:
            model_(data_)

    mean = model_trace.trace.nodes[data_type + "/xsg"]["fn"].mean
    if normalize_size:
        _, sizes = np.unique(
            imputation_label[imputation_label != 0].cpu().numpy(),
            return_counts=True,
        )
        mean = mean / torch.as_tensor(sizes).to(mean).unsqueeze(1)
    return mean


def compute_imputation(
    annotation_layer: str = "",
    num_samples: int = 1,
    normalize_size: bool = False,
):
    r"""Imputation analysis function"""
    dataloader = require("dataloader")
    save_path = require("save_path")

    output_dir = os.path.join(save_path, f"imputation-{annotation_layer}")

    with Session(
        default_device=torch.device("cpu"), pyro_stack=[]
    ), torch.no_grad():
        for name, slide in dataloader.dataset.data.slides.items():
            try:
                annotation = slide.data.annotation(annotation_layer)
            except RuntimeError:
                log(
                    WARNING,
                    'Slide "%s" does not have an annotation layer "%s"',
                    name,
                    annotation_layer,
                )
                continue

            samples = torch.stack(
                [
                    _impute(
                        slide,
                        dataloader.dataset.data.design[name],
                        torch.as_tensor(annotation.astype(np.int32)),
                        normalize_size=normalize_size,
                    )
                    for _ in tqdm(range(num_samples))
                ]
            )

            os.makedirs(
                os.path.join(output_dir, os.path.basename(name)), exist_ok=True
            )
            pd.concat(
                [
                    pd.DataFrame(
                        sample.detach().cpu().numpy(),
                        columns=slide.data.genes,
                        index=pd.Index(
                            np.unique(annotation)[1:], name="label"
                        ),
                    )
                    for sample in samples
                ],
                keys=pd.Index(np.arange(len(samples)) + 1, name="sample"),
            ).to_csv(
                os.path.join(
                    output_dir, os.path.basename(name), "imputed_counts.csv"
                )
            )


_register_analysis(
    name="imputation",
    analysis=Analysis(
        description="Imputes expression data", function=compute_imputation
    ),
)
