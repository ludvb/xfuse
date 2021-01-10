r"""Config file for tests"""

import itertools as it

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as distr
import pytest
import torch
from scipy.ndimage import label as make_label
from xfuse.convert.utility import write_data
from xfuse.data import Data, Dataset
from xfuse.data.slide import STSlide, FullSlideIterator, Slide
from xfuse.data.utility.misc import make_dataloader
from xfuse.model import XFuse
from xfuse.model.experiment.st import ST, MetageneDefault
from xfuse.session import Session, get
from xfuse.train import train
from xfuse.utility.state import reset_state


def pytest_configure(config):
    # pylint: disable=missing-function-docstring
    config.addinivalue_line(
        "markers", "fix_rng: resets the RNG to a fixed value"
    )
    config.addinivalue_line("markers", "slow: marks test as slow to run")


def pytest_runtest_setup(item):
    # pylint: disable=missing-function-docstring
    pyro.clear_param_store()
    reset_state()
    if item.get_closest_marker("fix_rng") is not None:
        torch.manual_seed(0)


def pytest_addoption(parser):
    # pylint: disable=missing-function-docstring
    parser.addoption(
        "--quick", action="store_true", default=False, help="skip slow tests"
    )


def pytest_collection_modifyitems(config, items):
    # pylint: disable=missing-function-docstring
    if config.getoption("--quick"):
        for item in filter(lambda x: "slow" in x.keywords, items):
            item.add_marker(pytest.mark.skip(reason="skipping slow test"))


@pytest.fixture
@pytest.mark.fix_rng
def toydata(tmp_path):
    r"""Produces toy dataset"""
    # pylint: disable=too-many-locals

    num_genes = 10
    num_metagenes = 3
    probs = 0.1
    H, W = [100] * 2
    spot_size = 10

    gridy, gridx = np.meshgrid(
        np.linspace(0.0, H - 1, H), np.linspace(0.0, W - 1, W)
    )
    yoffset, xoffset = (
        distr.Normal(0.0, 0.2).sample([2, num_metagenes]).cpu().numpy()
    )
    activity = (
        np.cos(gridy[..., None] / 100 - 0.5 + yoffset[None, None]) ** 2
        * np.cos(gridx[..., None] / 100 - 0.5 + xoffset[None, None]) ** 2
    )
    activity = torch.as_tensor(activity, dtype=torch.float32)

    metagene_profiles = (
        distr.Normal(0.0, 1.0)
        .expand([num_genes, num_metagenes])
        .sample()
        .exp()
    )

    label = np.zeros(activity.shape[:2]).astype(np.uint8)
    counts = [torch.zeros(num_genes)]
    for i, (y, x) in enumerate(
        it.product(
            (np.linspace(0.0, 1, H // spot_size)[1:-1] * H).astype(int),
            (np.linspace(0.0, 1, W // spot_size)[1:-1] * W).astype(int),
        ),
        1,
    ):
        spot_activity = torch.zeros(num_metagenes)

        for dy, dx in [
            (dx, dy)
            for dx, dy in (
                (dy - spot_size // 2, dx - spot_size // 2)
                for dy in range(spot_size)
                for dx in range(spot_size)
            )
            if dy ** 2 + dx ** 2 < spot_size ** 2 / 4
        ]:
            label[y + dy, x + dx] = i
            spot_activity += activity[y + dy, x + dx]
        rate = spot_activity @ metagene_profiles.t()
        counts.append(distr.NegativeBinomial(rate, probs).sample())

    image = 255 * (
        (activity - activity.min()) / (activity.max() - activity.min())
    )
    image = image.round().byte().cpu().numpy()
    counts = torch.stack(counts)
    counts = pd.DataFrame(
        counts.cpu().numpy(),
        index=pd.Index(list(range(counts.shape[0]))),
        columns=[f"g{i + 1}" for i in range(counts.shape[1])],
    )

    annotation1 = np.arange(100) // 10 % 2 == 1
    annotation1 = annotation1[:, None] & annotation1[None]
    annotation1, _ = make_label(annotation1)
    annotation2 = 1 + (annotation1 == 0).astype(np.uint8)

    filepath = tmp_path / "data.h5"
    write_data(
        counts,
        image,
        label,
        type_label="ST",
        annotation={
            "annotation1": (
                annotation1,
                {x: str(x) for x in np.unique(annotation1) if x != 0},
            ),
            "annotation2": (annotation2, {1: "false", 2: "true"}),
        },
        auto_rotate=True,
        path=str(filepath),
    )

    design = pd.DataFrame({"ID": 1}, index=["toydata"]).astype("category")
    slide = Slide(data=STSlide(str(filepath)), iterator=FullSlideIterator)
    data = Data(slides={"toydata": slide}, design=design)
    dataset = Dataset(data)
    dataloader = make_dataloader(dataset)

    return dataloader


@pytest.fixture
def pretrained_toy_model(toydata):
    r"""Pretrained toy model"""
    # pylint: disable=redefined-outer-name
    st_experiment = ST(
        depth=2,
        num_channels=4,
        metagenes=[MetageneDefault(0.0, None) for _ in range(1)],
    )
    xfuse = XFuse(experiments=[st_experiment])
    with Session(
        model=xfuse,
        optimizer=pyro.optim.Adam({"lr": 0.001}),
        dataloader=toydata,
        genes=toydata.dataset.genes,
        covariates={
            covariate: values.cat.categories.values.tolist()
            for covariate, values in toydata.dataset.data.design.iteritems()
        },
    ):
        train(100 + get("training_data").epoch)
    return xfuse
