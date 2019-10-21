#!/usr/bin/env python3

r"""writes hssl/__version__ to correspond to the current git version"""

import os.path as osp
import re
import subprocess as sp


def _output_from(cmd):
    return (
        sp.run(cmd.split(" "), capture_output=True, check=True)
        .stdout.decode()
        .strip()
    )


try:
    VERSION = _output_from("git describe --dirty")
    # Modify according to PEP440
    VERSION = (
        re.compile(r"([^\-]+?)-([^\-]+?)-(.+?)")
        .sub(r"\g<1>.post\g<2>+\g<3>", VERSION)
        .replace("-", ".")
    )
except sp.CalledProcessError:
    try:
        VERSION = _output_from("git describe --dirty --always")
        VERSION = "0.0.0+untagged.{}".format(VERSION.replace("-", "."))
    except sp.CalledProcessError:
        VERSION = "0.0.0+git.error"
except FileNotFoundError:
    VERSION = "0.0.0+no.git"

try:
    DIFF = _output_from("git diff")
except (FileNotFoundError, sp.CalledProcessError):
    DIFF = ""

with open(osp.join(osp.dirname(__file__), "hssl", "__version__.py"), "w") as f:
    f.write(
        "".join(
            map(
                lambda x: x + "\n",
                [
                    'r"""Generated automatically---don\'t edit!"""',
                    "",
                    f'__version__ = "{VERSION:s}"',
                    '__diff__ = r"""' + DIFF.replace('"', '\\"') + '"""',
                ],
            )
        )
    )
