import os
import warnings
from functools import partial, reduce
from operator import add
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ._config import _ANNOTATED_CONFIG as CONFIG  # type: ignore
from .analyze import analyses as _analyses
from .data import Data, Dataset
from .data.slide import RandomSlide, Slide, STSlide
from .data.utility.misc import make_dataloader
from .logging import INFO, log
from .model import XFuse
from .model.experiment.st import ST as STExperiment
from .model.experiment.st.metagene_expansion_strategy import (
    STRATEGIES,
    ExpansionStrategy,
)
from .model.experiment.st.metagene_eval import MetagenePurger, purge_metagenes
from .model.experiment.st.metagene_expansion_strategy import Extra
from .optim import Adam  # type: ignore  # pylint: disable=no-name-in-module
from .session import Session, get, require
from .train import test_convergence, train
from .utility.file import first_unique_filename
from .session.io import save_session


def run(
    design: pd.DataFrame,
    slide_paths: Dict[str, str],
    analyses: Dict[str, Dict[str, Any]] = None,
    expansion_strategy: ExpansionStrategy = STRATEGIES[
        CONFIG["expansion_strategy"].value["type"].value
    ](),
    purge_interval: int = CONFIG["expansion_strategy"]
    .value["purge_interval"]
    .value,
    network_depth: int = CONFIG["xfuse"].value["network_depth"].value,
    network_width: int = CONFIG["xfuse"].value["network_width"].value,
    min_counts: int = CONFIG["xfuse"].value["min_counts"].value,
    patch_size: int = CONFIG["optimization"].value["patch_size"].value,
    batch_size: int = CONFIG["optimization"].value["batch_size"].value,
    epochs: int = CONFIG["optimization"].value["epochs"].value,
    learning_rate: float = CONFIG["optimization"].value["learning_rate"].value,
    cache_data: bool = CONFIG["settings"].value["cache_data"].value,
    num_data_workers: int = CONFIG["settings"].value["data_workers"].value,
    slide_options: Optional[Dict[str, Any]] = None,
):
    r"""Runs an analysis"""

    # pylint: disable=too-many-arguments,too-many-locals

    if analyses is None:
        analyses = {}

    if (available_cores := len(os.sched_getaffinity(0))) < num_data_workers:
        warnings.warn(
            " ".join(
                [
                    f"Available cores ({available_cores}) is less than the"
                    f" requested number of workers ({num_data_workers}).",
                    f" Setting the number of workers to {available_cores}.",
                ]
            ),
        )
        num_data_workers = available_cores

    slides = {
        slide: Slide(
            data=STSlide(
                slide_paths[slide],
                cache_data=cache_data,
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
    dataloader = make_dataloader(
        dataset,
        batch_size=batch_size if batch_size < len(dataset) else len(dataset),
        shuffle=True,
        num_workers=num_data_workers,
        drop_last=True,
    )

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
            depth=network_depth, num_channels=network_width,
        )
        xfuse = XFuse(experiments=[st_experiment]).to(get("default_device"))

    optimizer = get("optimizer")
    if optimizer is None:
        optimizer = Adam({"amsgrad": True})

    def _panic(_session, _err_type, _err, _tb):
        save_session("exception")

    with Session(
        model=xfuse,
        genes=genes,
        learning_rate=learning_rate,
        messengers=[
            MetagenePurger(
                period=lambda e: (
                    purge_interval > 0
                    and e % purge_interval == 0
                    and (epochs < 0 or e <= epochs - purge_interval)
                ),
                num_samples=3,
            ),
            *get("messengers"),
        ],
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
            with Session(model=xfuse, metagene_expansion_strategy=Extra(0)):
                try:
                    purge_metagenes(num_samples=10)
                except RuntimeError as exc:
                    if "Cannot remove last metagene" in str(exc):
                        warnings.warn("Failed to find metagenes")
                    else:
                        raise
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
                warnings.warn(f'Unknown analysis "{name}"')
