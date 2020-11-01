import os

from xfuse.analyze.differential_expression import (
    compute_differential_expression,
)
from xfuse.session import Session
from xfuse.utility.design import extract_covariates


def test_compute_differential_expression(
    pretrained_toy_model, toydata, tmp_path
):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        covariates=extract_covariates(toydata.dataset.data.design),
        save_path=tmp_path,
        eval=True,
    ):
        compute_differential_expression("annotation1", "annotation2")

    assert os.path.exists(tmp_path / "differential_expression" / "data.csv.gz")
