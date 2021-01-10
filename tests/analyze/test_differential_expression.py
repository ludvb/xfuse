import os

from xfuse.analyze.differential_expression import (
    _run_differential_expression_analysis,
)
from xfuse.session import Session
from xfuse.session.items.work_dir import WorkDir


def test_run_differential_expression_analysis(
    pretrained_toy_model, toydata, tmp_path
):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        work_dir=WorkDir(tmp_path),
        eval=True,
    ):
        _run_differential_expression_analysis(
            "annotation2", comparisons=[("true", "false")]
        )

    assert os.path.exists(tmp_path / "true-vs-false.csv.gz")
    assert os.path.exists(tmp_path / "true-vs-false_top_differential.pdf")
