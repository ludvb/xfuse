import os

import numpy as np
import pandas as pd
from xfuse.analyze.prediction import _run_prediction_analysis
from xfuse.session import Session
from xfuse.session.items.work_dir import WorkDir


def test_run_prediction_analysis(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        work_dir=WorkDir(tmp_path),
        eval=True,
    ):
        _run_prediction_analysis("annotation1")

    for name, slide in toydata.dataset.data.slides.items():
        name = os.path.basename(name)
        output_file = tmp_path / "data.csv.gz"
        assert os.path.exists(output_file)

        output_data = pd.read_csv(output_file)
        output_data_labels = list(np.unique(output_data.annotation1))
        _, annotation_labels = slide.data.annotation("annotation1")
        annotation_labels = sorted(annotation_labels.keys())
        assert output_data_labels == annotation_labels
