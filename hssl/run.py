from functools import partial
import os
from typing import Any, Dict, Optional

import h5py
import pandas as pd
import pyro

from ._config import _ANNOTATED_CONFIG as CONFIG  # type: ignore
from .data import Data, Dataset
from .data.slide import RandomSlide, Slide, STSlide
from .data.utility.misc import make_dataloader, spot_size
from .model import XFuse
from .model.experiment.st import ST as STExperiment
from .model.experiment.st.factor_expansion_strategy import (
    STRATEGIES,
    ExpansionStrategy,
)
from .model.experiment.st import (
    ExtraBaselines,
    FactorDefault,
    purge_factors,
)
from .session import Session, Unset, get
from .train import train
from .utility.session import save_session


def run(
    design: pd.DataFrame,
    expansion_strategy: ExpansionStrategy = STRATEGIES[
        CONFIG["expansion_strategy"].value["type"].value
    ](),
    network_depth: int = CONFIG["xfuse"].value["network_depth"].value,
    network_width: int = CONFIG["xfuse"].value["network_width"].value,
    patch_size: int = CONFIG["optimization"].value["patch_size"].value,
    batch_size: int = CONFIG["optimization"].value["batch_size"].value,
    epochs: int = CONFIG["optimization"].value["epochs"].value,
    learning_rate: float = CONFIG["optimization"].value["learning_rate"].value,
    slide_options: Optional[Dict[str, Any]] = None,
):
    r"""Runs an analysis"""
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

    xfuse = get("model")
    if xfuse is None:
        st_experiment = STExperiment(
            depth=network_depth,
            num_channels=network_width,
            factors=[FactorDefault(0.0, None)],
            default_scale=1 / spot_size(dataset),
        )
        xfuse = XFuse(experiments=[st_experiment]).to(get("default_device"))

    optimizer = get("optimizer")
    if optimizer is None:
        optimizer = pyro.optim.ClippedAdam(
            {"lr": learning_rate, "weight_decay": 1e-5}
        )

    def _panic(_session, _err_type, _err, _tb):
        with Session(dataloader=Unset, panic=Unset, pyro_stack=[]):
            save_session(f"exception")

    with Session(
        model=xfuse,
        factor_expansion_strategy=expansion_strategy,
        optimizer=optimizer,
        dataloader=dataloader,
        panic=_panic,
    ):
        train(epochs)

        with Session(factor_expansion_strategy=ExtraBaselines(0)):
            purge_factors(xfuse, num_samples=10)

        with Session(dataloader=Unset, panic=Unset, pyro_stack=[]):
            save_session(f"final")
