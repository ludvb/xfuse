import pyro
from xfuse.model.experiment.st.st import _encode_metagene_name
from xfuse.session import Session, get
from xfuse.utility.tensor import to_device


def test_split_metagene(pretrained_toy_model, toydata):
    r"""Test that metagenes are split correctly"""
    st_experiment = pretrained_toy_model.get_experiment("ST")
    metagene = next(iter(st_experiment.metagenes.keys()))
    metagene_new = st_experiment.split_metagene(metagene)

    with Session(
        model=pretrained_toy_model,
        dataloader=toydata,
        genes=toydata.dataset.genes,
        covariates={
            covariate: values.cat.categories.values.tolist()
            for covariate, values in toydata.dataset.data.design.iteritems()
        },
    ):
        x = to_device(next(iter(toydata)))
        with pyro.poutine.trace() as guide_tr:
            get("model").guide(x)
        with pyro.poutine.trace() as model_tr:
            with pyro.poutine.replay(trace=guide_tr.trace):
                get("model").model(x)

    rim_mean = model_tr.trace.nodes["rim"]["fn"].mean
    assert (rim_mean[0, 0] == rim_mean[-1][0, -1]).all()

    rate_mg = guide_tr.trace.nodes[_encode_metagene_name(metagene)]["fn"].mean
    rate_mg_new = guide_tr.trace.nodes[_encode_metagene_name(metagene_new)][
        "fn"
    ].mean
    assert (rate_mg == rate_mg_new).all()
