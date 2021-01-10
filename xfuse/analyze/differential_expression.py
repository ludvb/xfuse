import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import Data, Dataset
from ..data.utility.misc import make_dataloader
from ..data.slide import AnnotatedImage, FullSlideIterator, Slide
from ..session import Session, require
from .analyze import Analysis, _register_analysis
from .prediction import predict_df


def _run_differential_expression_analysis(
    annotation_layer: Optional[str] = None,
    comparisons: List[Tuple[str, str]] = None,
    normalize_covariates: List[str] = None,
    num_samples: int = 50,
    genes_per_batch: int = 100000,
) -> None:
    """Runs differential expression analysis"""

    dataloader = require("dataloader")

    if annotation_layer is None:
        warnings.warn(
            "No annotation layer specified."
            " Skipping differential gene expression analysis."
        )
        return

    if comparisons is None:
        warnings.warn(
            "No comparisons specified."
            " Skipping differential gene expression analysis."
        )
        return

    if normalize_covariates is None:
        normalize_covariates = []

    slides = {
        slide_name: Slide(
            data=AnnotatedImage.from_st_slide(
                slide.data, annotation_name=annotation_layer
            ),
            iterator=FullSlideIterator,
        )
        for slide_name, slide in dataloader.dataset.data.slides.items()
    }
    dataloader = make_dataloader(
        Dataset(Data(slides, design=dataloader.dataset.data.design)),
        batch_size=1,
        shuffle=False,
    )

    with Session(dataloader=dataloader, messengers=[]):
        samples = predict_df(
            num_samples=num_samples,
            genes_per_batch=genes_per_batch,
            normalize_covariates=normalize_covariates,
        )

    samples = samples.groupby([annotation_layer, "sample", "gene"]).agg(sum)
    samples = samples.assign(
        count=samples.groupby([annotation_layer, "sample"]).transform(
            lambda x: np.log2(x / x.sum())
        )
    )
    samples = samples.reset_index().pivot(
        ["sample", "gene"], columns=annotation_layer
    )
    samples.columns = samples.columns.map(lambda x: x[1])

    def _save_comparison(a, b):
        lfc = (
            samples[[a, b]]
            .assign(lfc=lambda x: x[a] - x[b])["lfc"]
            .reset_index()
            .pivot("sample", "gene")
        )
        lfc.columns = lfc.columns.map(lambda x: x[1])

        lfc.to_csv(f"{a}-vs-{b}.csv.gz")

        sorted_values = lfc.mean(0).sort_values()
        log2_fold_top = lfc[
            pd.concat([sorted_values[:10], sorted_values[-10:]]).index
        ]
        log2_fold_top.boxplot(vert=False)
        plt.title(f"{a} vs. {b}")
        plt.xlabel("log2 fold")
        plt.savefig(f"{a}-vs-{b}_top_differential.pdf")
        plt.close()

    for a, b in comparisons:
        _save_comparison(a, b)


_register_analysis(
    name="differential_expression",
    analysis=Analysis(
        description="Performs differential gene expression analysis",
        function=_run_differential_expression_analysis,
    ),
)
