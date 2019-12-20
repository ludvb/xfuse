import os
from typing import cast

import numpy as np
import pandas as pd
import pyro
import torch
from pyro.poutine.messenger import Messenger
from tifffile import imwrite

from ..logging import WARNING, log
from ..model.experiment.st.st import ST, _encode_factor_name
from ..session import Session, require
from ..utility.visualization import visualize_factors
from .analyze import Analysis, _register_analysis

__all__ = ["compute_metagene_summary"]


class __ActivationTracker(Messenger):
    # pylint: disable=invalid-name
    def __init__(self):
        super().__init__()
        self.activations = []

    def _pyro_post_sample(self, msg):
        if not (msg["type"] == "sample" and msg["name"][-3:] == "rim"):
            return
        self.activations.append(msg["fn"].mean.squeeze())


def _normalize(image: np.ndarray) -> np.ndarray:
    minvals = image.min((0, 1))
    maxvals = image.max((0, 1))
    return (image - minvals) / (maxvals - minvals)


def compute_metagene_summary(method: str = "pca") -> None:
    r"""Imputation analysis function"""
    # pylint: disable=too-many-locals
    dataloader = require("dataloader")
    model = require("model")
    save_path = require("save_path")

    output_dir = os.path.join(save_path, f"metagenes")

    def _metagene_profile_st():
        experiment = cast(ST, model.get_experiment("ST"))
        with pyro.poutine.trace() as trace:
            # pylint: disable=protected-access
            experiment._sample_globals()
        return [
            trace.trace.nodes[_encode_factor_name(n)]["fn"]
            for n in experiment.factors
        ]

    metagene_profile_fn = {
        "ST": _metagene_profile_st,
    }

    with Session(
        default_device=torch.device("cpu"), pyro_stack=[]
    ), torch.no_grad():
        with __ActivationTracker() as activation_tracker:
            for slide_path, summarization, factors in zip(
                dataloader.dataset.data.design.columns,
                visualize_factors(method),
                activation_tracker.activations,
            ):
                slide_name = os.path.basename(slide_path)
                os.makedirs(
                    os.path.join(output_dir, slide_name), exist_ok=True,
                )
                imwrite(
                    os.path.join(output_dir, slide_name, "summary.png"),
                    summarization,
                )
                mask = summarization.sum(-1) != 0.0
                for n, factor in enumerate(factors, 1):
                    factor = -factor.detach().cpu().numpy()
                    factor[~mask] = factor[mask].max()
                    factor = _normalize(factor)
                    imwrite(
                        os.path.join(
                            output_dir, slide_name, f"factor-{n}.png"
                        ),
                        factor,
                    )

        for experiment in model.experiments.keys():
            try:
                metagene_profiles = metagene_profile_fn[experiment]()
            except KeyError:
                log(
                    WARNING,
                    'Metagene profiles for experiment of type "%s" '
                    " not implemented",
                    experiment,
                )
                continue
            metagene_profiles_ = (
                pd.concat(
                    [
                        pd.DataFrame(
                            [
                                x.mean.detach().cpu().numpy(),
                                x.stddev.detach().cpu().numpy(),
                            ],
                            columns=dataloader.dataset.genes,
                            index=pd.Index(["mean", "stddev"], name="type"),
                        )
                        for x in metagene_profiles
                    ],
                    keys=pd.Index(
                        np.arange(len(metagene_profiles)) + 1, name="metagene"
                    ),
                )
                .reset_index()
                .melt(
                    ["metagene", "type"],
                    var_name="gene",
                    value_name="log2fold",
                )
            )
            metagene_profiles_.to_csv(
                os.path.join(output_dir, f"{experiment}-metagenes.csv.gz"),
            )


_register_analysis(
    name="metagenes",
    analysis=Analysis(
        description="Creates summary data of the metagenes",
        function=compute_metagene_summary,
    ),
)
