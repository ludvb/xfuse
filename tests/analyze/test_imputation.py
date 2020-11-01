import os

import numpy as np
import pandas as pd
from xfuse.analyze.imputation import compute_imputation
from xfuse.session import Session
from xfuse.utility.design import extract_covariates


def test_compute_imputation(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        covariates=extract_covariates(toydata.dataset.data.design),
        save_path=tmp_path,
        eval=True,
    ):
        compute_imputation("annotation1")

    for name, slide in toydata.dataset.data.slides.items():
        name = os.path.basename(name)
        output_file = (
            tmp_path / "imputation-annotation1" / name / "imputed_counts.csv"
        )
        assert os.path.exists(output_file)

        output_data = pd.read_csv(output_file)
        output_data_labels = np.unique(output_data.label)
        output_data_labels = np.sort(output_data_labels)
        annotation_labels = np.unique(slide.data.annotation("annotation1"))
        annotation_labels = np.sort(annotation_labels[annotation_labels > 0])
        assert (annotation_labels == output_data_labels).all()
