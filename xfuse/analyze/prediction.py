import itertools as it
from copy import copy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyro
import torch

from ..data import Data, Dataset
from ..data.slide import AnnotatedImage, FullSlideIterator, Slide
from ..data.utility.misc import make_dataloader
from ..logging import Progressbar
from ..session import Session, require
from ..utility.core import chunks_of
from ..utility.tensor import to_device
from .analyze import Analysis, _register_analysis


def _run_model(data, normalize_covariates, normalize_size, predict_mean):
    genes = require("genes")
    model = require("model")

    data_type, *__this_should_be_empty = list(data.keys())

    if data_type != "AnnotatedImage":
        raise ValueError(
            f'Invalid data type "{data_type}".'
            " Can only predict on AnnotatedImage."
        )
    assert __this_should_be_empty == []

    data = to_device(data[data_type])

    # pylint: disable=fixme
    # FIXME: Compatibility hack.
    #        Let's get back to this when reworking the model code.
    st_data = {
        "slide": data["slide"],
        "covariates": [
            {
                covariate: condition
                for covariate, condition in covariates.items()
                if covariate not in normalize_covariates
            }
            for covariates in data["covariates"]
        ],
        "data": [
            to_device(
                torch.zeros(
                    int(label.max().item()), len(genes), dtype=torch.float32,
                )
            )
            for label in data["label"]
        ],
        "label": data["label"],
        "image": data["image"],
    }

    with Session(eval=True):
        with pyro.poutine.trace() as guide_trace:
            model.guide({"ST": st_data})

    with Session(eval=True):
        with pyro.poutine.replay(trace=guide_trace.trace):
            with pyro.poutine.trace() as model_trace:
                model({"ST": st_data})

    for i, (annotation_name, label, label_names, section_name) in enumerate(
        zip(data["name"], data["label"], data["label_names"], data["slide"],)
    ):
        try:
            idxs = (
                model_trace.trace.nodes[f"ST/idx-{i}"]["value"]
                .long()
                .cpu()
                .numpy()
            )
            emission_distr = model_trace.trace.nodes[f"ST/xsg-{i}"]["fn"]
        except KeyError:
            continue

        sample = (
            emission_distr.mean if predict_mean else emission_distr.sample()
        )

        if normalize_size:
            _, sizes = np.unique(
                np.searchsorted(idxs, label.cpu().numpy()), return_counts=True,
            )
            sample = sample / torch.as_tensor(sizes).to(sample).unsqueeze(1)

        label_names = label_names[idxs]

        yield {
            "data": sample,
            "rownames": label_names,
            "colnames": genes,
            "section": section_name,
            "annotation": annotation_name,
        }


def predict(
    num_samples: int = 1,
    genes_per_batch: int = 10,
    predict_mean: bool = True,
    normalize_scale: bool = False,
    normalize_size: bool = False,
    normalize_covariates: Optional[List[str]] = None,
) -> Iterable[Dict[str, Any]]:
    """Predicts gene expression"""

    if normalize_covariates is None:
        normalize_covariates = []

    dataloader = require("dataloader")
    genes = require("genes")
    model = require("model")

    def _sample():
        conditional_model = copy(model)
        if normalize_scale:
            conditional_model.model = pyro.poutine.condition(
                conditional_model.model,
                # pylint: disable=not-callable
                {"scale": torch.tensor(1.0)},
            )

        iterator = it.product(dataloader, chunks_of(genes, genes_per_batch))

        with pyro.poutine.trace() as global_trace:
            with pyro.poutine.block(
                expose_fn=lambda msg: (
                    "is_guide" in msg
                    and msg["is_guide"]
                    and "is_global" in msg["infer"]
                    and msg["infer"]["is_global"]
                )
            ):
                try:
                    data, batch_genes = next(iterator)
                except StopIteration:
                    return
                with Session(genes=batch_genes, model=conditional_model):
                    yield from _run_model(
                        data=data,
                        predict_mean=predict_mean,
                        normalize_size=normalize_size,
                        normalize_covariates=normalize_covariates,
                    )

        conditional_model.guide = pyro.poutine.condition(
            conditional_model.guide,
            {
                variable: properties["value"]
                for variable, properties in global_trace.trace.nodes.items()
            },
        )
        conditional_model.model = pyro.poutine.condition(
            conditional_model.model,
            {
                variable: properties["value"]
                for variable, properties in global_trace.trace.nodes.items()
            },
        )

        for data, batch_genes in iterator:
            with Session(genes=batch_genes, model=conditional_model):
                yield from _run_model(
                    data=data,
                    predict_mean=predict_mean,
                    normalize_size=normalize_size,
                    normalize_covariates=normalize_covariates,
                )

    with Progressbar(
        range(1, num_samples + 1), desc="Sampling", leave=False
    ) as iterator:
        for sample_num in iterator:
            for sample in _sample():
                yield {**sample, "sample": sample_num}


def predict_df(**kwargs) -> pd.DataFrame:
    """
    Similar to :func:`predict` but, instead of streaming result :class:`Dict`s,
    return all results in a tidy :class:`~pd.DataFrame`.
    """
    return pd.concat(
        [
            pd.DataFrame(x["data"].cpu().numpy(), columns=x["colnames"])
            .assign(
                **{
                    x["annotation"]: x["rownames"],
                    "section": x["section"],
                    "sample": x["sample"],
                }
            )
            .melt(
                [x["annotation"], "section", "sample"],
                var_name="gene",
                value_name="count",
            )
            for x in predict(**kwargs)
        ],
        axis=0,
    )


def _run_prediction_analysis(
    annotation_layer: str = "",
    num_samples: int = 1,
    genes_per_batch: int = 10,
    predict_mean: bool = True,
    normalize_scale: bool = False,
    normalize_size: bool = False,
    normalize_covariates: Optional[List[str]] = None,
) -> None:
    """Runs prediction analysis"""

    if normalize_covariates is None:
        normalize_covariates = []

    dataloader = require("dataloader")
    dataloader = make_dataloader(
        Dataset(
            Data(
                slides={
                    k: Slide(
                        data=AnnotatedImage.from_st_slide(
                            v.data, annotation_name=annotation_layer
                        ),
                        iterator=FullSlideIterator,
                    )
                    for k, v in dataloader.dataset.data.slides.items()
                },
                design=dataloader.dataset.data.design,
            )
        ),
        batch_size=1,
        shuffle=False,
    )

    with Session(dataloader=dataloader, messengers=[]):
        samples = predict_df(
            num_samples=num_samples,
            genes_per_batch=genes_per_batch,
            predict_mean=predict_mean,
            normalize_scale=normalize_scale,
            normalize_size=normalize_size,
            normalize_covariates=normalize_covariates,
        )

    samples = samples.pivot(
        index=[annotation_layer, "section", "sample"], columns=["gene"]
    )
    samples = samples.reset_index()
    samples.columns = samples.columns.map(
        lambda x: x[1] if x[1] != "" else x[0]
    )
    samples.to_csv("data.csv.gz")


_register_analysis(
    name="prediction",
    analysis=Analysis(
        description="Predicts expression data",
        function=_run_prediction_analysis,
    ),
)
