import warnings
from copy import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from tqdm import tqdm

from ..session import Session, require
from .analyze import Analysis, _register_analysis
from .imputation import _impute


def compute_differential_expression(
    annotation_layer1: str = "",
    annotation_layer2: str = "",
    num_samples: int = 10,
):
    r"""Imputation analysis function"""
    # pylint: disable=too-many-locals
    dataloader = require("dataloader")
    model = require("model")

    with Session(
        default_device=torch.device("cpu"), messengers=[]
    ), torch.no_grad():
        samples: Dict[str, List[torch.Tensor]] = {
            annotation_layer1: [],
            annotation_layer2: [],
        }
        for _ in tqdm(range(num_samples), position=1):
            model_fixed_globals = None
            for slide_name, slide in tqdm(
                dataloader.dataset.data.slides.items(), position=0
            ):
                for annotation_name in (annotation_layer1, annotation_layer2):
                    try:
                        annotation_label = torch.as_tensor(
                            slide.data.annotation(annotation_name) != 0
                        )
                    except RuntimeError:
                        warnings.warn(
                            f'Slide "{slide_name}" does not have an annotation'
                            f' layer "{annotation_name}"',
                        )
                        continue

                    design = dataloader.dataset.data.design[slide_name]

                    # Get model trace, retaining globals from the first sample
                    # NOTE: This assumes a conditioning structure
                    #    q(local, global) = q(global)\Pi_i q(local_i|global).
                    if model_fixed_globals:
                        with Session(model=model_fixed_globals):
                            sample = _impute(
                                slide,
                                design,
                                annotation_label,
                                normalize_scale=False,
                                normalize_size=False,
                            )
                    else:
                        with pyro.poutine.trace() as trace:
                            with pyro.poutine.block(
                                hide_fn=lambda msg: "is_guide" in msg
                                and msg["is_guide"]
                            ):
                                sample = _impute(
                                    slide,
                                    design,
                                    annotation_label,
                                    normalize_scale=False,
                                    normalize_size=False,
                                )
                        model_fixed_globals = copy(model)
                        model_fixed_globals.model = pyro.poutine.condition(
                            model_fixed_globals.model,
                            {
                                k: v["value"]
                                for k, v in trace.trace.nodes.items()
                                if "is_global" in v["infer"]
                                if v["infer"]["is_global"]
                            },
                        )
                    samples[annotation_name].append(sample.squeeze())

    annotation_samples1 = torch.stack(samples[annotation_layer1])
    annotation_samples2 = torch.stack(samples[annotation_layer2])

    annotation_samples1 = (
        annotation_samples1
        / annotation_samples1.sum(  # type: ignore
            1, keepdims=True
        )
    )
    annotation_samples2 = (
        annotation_samples2
        / annotation_samples2.sum(  # type: ignore
            1, keepdims=True
        )
    )

    log2_fold = annotation_samples1.log2() - annotation_samples2.log2()
    data = pd.DataFrame(
        log2_fold.cpu().numpy(),
        index=pd.Index(np.arange(num_samples) + 1, name="sample"),
        columns=dataloader.dataset.genes,
    )

    data.to_csv("data.csv.gz", index=False)

    sorted_values = data.mean(0).sort_values()
    log2_fold_top = data[
        pd.concat([sorted_values[:10], sorted_values[-10:]]).index
    ]
    log2_fold_top.boxplot(vert=False)
    plt.title(f"{annotation_layer1} vs. {annotation_layer2}")
    plt.xlabel("log2 fold")
    plt.savefig("top_differential.pdf")


_register_analysis(
    name="differential_expression",
    analysis=Analysis(
        description="Performs differential gene expression analysis",
        function=compute_differential_expression,
    ),
)
