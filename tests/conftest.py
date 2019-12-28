r"""Config file for tests"""

import itertools as it

import h5py
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as distr
import torch

import pytest
from xfuse.convert.utility import write_data
from xfuse.data import Data, Dataset
from xfuse.data.slide import STSlide, FullSlide, Slide
from xfuse.data.utility.misc import make_dataloader
from xfuse.utility import design_matrix_from
from xfuse.utility.modules import reset_state


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

    filepath = tmp_path / "data.h5"
    write_data(counts, image, label, {}, "ST", str(filepath))

    design_matrix = design_matrix_from({str(filepath): {"ID": 1}})
    slide = Slide(data=STSlide(h5py.File(filepath, "r")), iterator=FullSlide)
    data = Data(slides={str(filepath): slide}, design=design_matrix)
    dataset = Dataset(data, unify_genes=True)
    dataloader = make_dataloader(dataset)

    return dataloader
