import os

from xfuse.analyze.gene_maps import compute_gene_maps
from xfuse.session import Session


def test_gene_maps(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        save_path=tmp_path,
        eval=True,
    ):
        compute_gene_maps()

    for section in map(os.path.basename, toydata.dataset.data.design):
        for gene in toydata.dataset.genes:
            assert os.path.exists(
                tmp_path / "gene_maps" / section / f"{gene}.png"
            )
