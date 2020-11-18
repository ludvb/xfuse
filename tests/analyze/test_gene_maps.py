import os

from xfuse.analyze.gene_maps import compute_gene_maps
from xfuse.session import Session
from xfuse.utility.design import extract_covariates


def test_gene_maps(pretrained_toy_model, toydata, tmp_path):
    with Session(
        model=pretrained_toy_model,
        genes=toydata.dataset.genes,
        dataloader=toydata,
        covariates=extract_covariates(toydata.dataset.data.design),
        save_path=tmp_path,
        eval=True,
    ):
        compute_gene_maps()

    for section in toydata.dataset.data.design:
        for gene in toydata.dataset.genes:
            assert os.path.exists(
                tmp_path / "gene_maps" / section / f"{gene}.png"
            )
