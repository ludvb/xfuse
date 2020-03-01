import os

from xfuse.analyze.differential_expression import (
    compute_differential_expression,
)
from xfuse.session import Session


def test_compute_differential_expression(
    pretrained_toy_model, toydata, tmp_path
):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        save_path=tmp_path,
        eval=True,
    ):
        compute_differential_expression("annotation1", "annotation2")

    assert os.path.exists(tmp_path / "differential_expression" / "data.csv.gz")
