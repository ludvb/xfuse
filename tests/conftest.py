""" Config file for tests
"""

import itertools as it

import numpy as np
import pyro
import pyro.distributions as distr
import pytest
import torch
from pyvips import Image


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fix_rng: resets the RNG to a fixed value"
    )
    config.addinivalue_line("markers", "slow: marks test as slow to run")


def pytest_runtest_setup(item):
    pyro.clear_param_store()
    if item.get_closest_marker("fix_rng") is not None:
        torch.manual_seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--quick", action="store_true", default=False, help="skip slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--quick"):
        for item in filter(lambda x: "slow" in x.keywords, items):
            item.add_marker(pytest.mark.skip(reason="skipping slow test"))


@pytest.fixture
@pytest.mark.fix_rng
def toydata():
    """Produces toy dataset"""
    # pylint: disable=too-many-locals

    num_genes = 10
    num_factors = 3
    probs = 0.1
    H, W = [100] * 2
    spot_size = 10

    gridy, gridx = np.meshgrid(
        np.linspace(0.0, H - 1, H), np.linspace(0.0, W - 1, W)
    )
    yoffset, xoffset = (
        distr.Normal(0.0, 0.2).sample([2, num_factors]).cpu().numpy()
    )
    activity = (
        np.cos(gridy[..., None] / 100 - 0.5 + yoffset[None, None]) ** 2
        * np.cos(gridx[..., None] / 100 - 0.5 + xoffset[None, None]) ** 2
    )
    activity = torch.as_tensor(activity, dtype=torch.float32)

    factor_profiles = (
        distr.Normal(0.0, 1.0).expand([num_genes, num_factors]).sample().exp()
    )

    label = np.zeros(activity.shape[:2]).astype(int)
    counts = [torch.zeros(num_genes)]
    for i, (y, x) in enumerate(
        it.product(
            (np.linspace(0.0, 1, H // spot_size)[1:-1] * H).astype(int),
            (np.linspace(0.0, 1, W // spot_size)[1:-1] * W).astype(int),
        ),
        1,
    ):
        spot_activity = torch.zeros(num_factors)

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
        rate = spot_activity @ factor_profiles.t()
        counts.append(distr.NegativeBinomial(rate, probs).sample())

    image = 255 * (
        (activity - activity.min()) / (activity.max() - activity.min())
    )
    image = image.round().byte()
    image = Image.new_from_memory(
        image.cpu().numpy().data,
        image.shape[1],
        image.shape[0],
        num_factors,
        "uchar",
    )

    label = Image.new_from_memory(
        # pylint: disable=unsubscriptable-object
        label.astype(np.uint8).data,
        label.shape[1],
        label.shape[0],
        1,
        "uchar",
    )

    return torch.stack(counts).float().to_sparse(), image, label
