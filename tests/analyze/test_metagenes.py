import os

from xfuse.analyze.metagenes import compute_metagene_summary
from xfuse.session import Session
from xfuse.session.items.work_dir import WorkDir


def test_metagenes(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        work_dir=WorkDir(tmp_path),
        eval=True,
    ):
        compute_metagene_summary()

    for section in toydata.dataset.data.slides:
        assert os.path.exists(tmp_path / section / f"summary.png")
