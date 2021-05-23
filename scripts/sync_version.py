#!/usr/bin/env python3

# pylint: disable=invalid-name

import os
import sys

import tomlkit


project_file = sys.argv[1]

with open(project_file, "r") as fp:
    project_config = tomlkit.loads(fp.read())

version = project_config["tool"]["poetry"]["version"]  # type: ignore

with open(
    os.path.join(os.path.dirname(project_file), "xfuse", "__version__.py"), "w"
) as fp:
    fp.write(f'__version__ = "{version}"\n')
