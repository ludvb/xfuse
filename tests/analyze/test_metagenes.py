import os

from xfuse.analyze.metagenes import compute_metagene_summary
from xfuse.session import Session


def test_metagenes(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        save_path=tmp_path,
        eval=True,
    ):
        compute_metagene_summary()

    for section in map(os.path.basename, toydata.dataset.data.design):
        assert os.path.exists(
            tmp_path / "metagenes" / section / f"summary.png"
        )
