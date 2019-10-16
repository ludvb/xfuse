#!/usr/bin/env python3

""" hssl setup script
"""

import os.path as osp
from subprocess import run

import setuptools as st

run(
    ["env", "python3", osp.join(osp.dirname(__file__), "refresh_version.py")],
    check=True,
)

st.setup()
