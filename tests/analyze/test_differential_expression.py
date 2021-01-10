import os

from xfuse.analyze.differential_expression import (
    compute_differential_expression,
)
from xfuse.session import Session
from xfuse.session.items.work_dir import WorkDir
from xfuse.utility.design import extract_covariates


def test_compute_differential_expression(
    pretrained_toy_model, toydata, tmp_path
):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        covariates=extract_covariates(toydata.dataset.data.design),
        work_dir=WorkDir(tmp_path),
        eval=True,
    ):
        compute_differential_expression("annotation1", "annotation2")

    assert os.path.exists(tmp_path / "data.csv.gz")
