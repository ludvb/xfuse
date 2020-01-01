from functools import partial, reduce
from operator import add
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import pyro
import torch

from ._config import _ANNOTATED_CONFIG as CONFIG  # type: ignore
from .analyze import analyses as _analyses
from .data import Data, Dataset
from .data.slide import RandomSlide, Slide, STSlide
from .data.utility.misc import make_dataloader
from .logging import INFO, WARNING, log
from .model import XFuse
from .model.experiment.st import ST as STExperiment
from .model.experiment.st.metagene_expansion_strategy import (
    STRATEGIES,
    ExpansionStrategy,
)
from .model.experiment.st import (
    ExtraBaselines,
    MetageneDefault,
    purge_metagenes,
)
from .session import Session, get, require
from .train import test_convergence, train
from .utility.file import first_unique_filename
from .utility.session import save_session


class __OptimizerStep:  # pylint: disable=invalid-name
    def __init__(self, warmup_epochs: int):
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        return min(1.0, epoch / self.warmup_epochs)


def run(
    design: pd.DataFrame,
    analyses: Dict[str, Dict[str, Any]] = None,
    expansion_strategy: ExpansionStrategy = STRATEGIES[
        CONFIG["expansion_strategy"].value["type"].value
    ](),
    network_depth: int = CONFIG["xfuse"].value["network_depth"].value,
    network_width: int = CONFIG["xfuse"].value["network_width"].value,
    encode_expression: bool = CONFIG["xfuse"].value["encode_expression"].value,
    genes: List[str] = CONFIG["xfuse"].value["genes"].value,
    min_counts: int = CONFIG["xfuse"].value["min_counts"].value,
    patch_size: int = CONFIG["optimization"].value["patch_size"].value,
    batch_size: int = CONFIG["optimization"].value["batch_size"].value,
    epochs: int = CONFIG["optimization"].value["epochs"].value,
    learning_rate: float = CONFIG["optimization"].value["learning_rate"].value,
    warmup_epochs: int = CONFIG["optimization"].value["warmup_epochs"].value,
    slide_options: Optional[Dict[str, Any]] = None,
):
    r"""Runs an analysis"""

    # pylint: disable=too-many-arguments,too-many-locals

    if analyses is None:
        analyses = {}

    slides = {
        slide: Slide(
            data=STSlide(
                h5py.File(os.path.expanduser(slide), "r"),
                **(slide_options[slide] if slide_options is not None else {}),
            ),
            iterator=partial(
                RandomSlide,
                patch_size=(
                    None if patch_size < 0 else (patch_size, patch_size)
                ),
            ),
        )
        for slide in design.columns
    }
    dataset = Dataset(data=Data(slides=slides, design=design))
    dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=True)

    genes = get("genes")
    if genes is None:
        summed_counts = reduce(
            add,
            [
                np.array(slide.data.counts.sum(0)).flatten()
                for slide in dataset.data.slides.values()
            ],
        )
        filtered_genes = set(
            g for g, x in zip(dataset.genes, summed_counts) if x < min_counts
        )
        if len(filtered_genes) > 0:
            log(
                INFO,
                "The following %d genes have less than %d counts and will"
                " therefore be excluded: %s",
                len(filtered_genes),
                min_counts,
                ", ".join(sorted(filtered_genes)),
            )
        genes = [g for g in dataset.genes if g not in filtered_genes]

    xfuse = get("model")
    if xfuse is None:
        st_experiment = STExperiment(
            depth=network_depth,
            num_channels=network_width,
            metagenes=[MetageneDefault(0.0, None)],
            encode_expression=encode_expression,
        )
        xfuse = XFuse(experiments=[st_experiment]).to(get("default_device"))

    optimizer = get("optimizer")
    if optimizer is None:
        optimizer = pyro.optim.LambdaLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": learning_rate, "amsgrad": True},
                "lr_lambda": __OptimizerStep(warmup_epochs),
            }
        )

    def _panic(_session, _err_type, _err, _tb):
        with Session(
            dataloader=None,
            default_device=None,
            log_file=None,
            panic=None,
            pyro_stack=[],
        ):
            save_session("exception")

    with Session(
        model=xfuse,
        genes=genes,
        metagene_expansion_strategy=expansion_strategy,
        optimizer=optimizer,
        dataloader=dataloader,
        panic=_panic,
    ):
        has_converged = (
            test_convergence()
            if epochs < 0
            else get("training_data").epoch >= epochs
        )
        if not has_converged:
            train(epochs)
            with Session(metagene_expansion_strategy=ExtraBaselines(0)):
                purge_metagenes(xfuse, num_samples=10)
            with Session(
                dataloader=None,
                default_device=None,
                log_file=None,
                panic=None,
                pyro_stack=[],
            ):
                save_session("final")

    with Session(
        model=xfuse,
        genes=genes,
        dataloader=dataloader,
        save_path=first_unique_filename(
            os.path.join(require("save_path"), "analyses")
        ),
        eval=True,
    ):
        for name, options in analyses.items():
            if name in analyses:
                log(INFO, 'Running analysis "%s"', name)
                _analyses[name].function(**options)
            else:
                log(WARNING, 'Unknown analysis "%s"', name)
