''' Config file for tests
'''

import pytest

import torch


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'fix_rng: resets the RNG to a fixed value')
    config.addinivalue_line(
        'markers',
        'slow: marks test as slow to run')


def pytest_runtest_setup(item):
    if item.get_closest_marker('fix_rng') is not None:
        torch.manual_seed(0)


def pytest_addoption(parser):
    parser.addoption(
        '--quick', action='store_true', default=False, help='skip slow tests')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--quick'):
        for item in filter(lambda x: 'slow' in x.keywords, items):
            item.add_marker(pytest.mark.skip(reason='skipping slow test'))
