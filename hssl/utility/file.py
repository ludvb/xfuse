import itertools as it

import os

from typing import Optional


def unique_prefix(prefix):
    for path in it.chain((prefix,), (f"{prefix}.{i}" for i in it.count(1))):
        if not os.path.exists(path):
            return path


def extension(path: str) -> Optional[str]:
    r""" Extract file name extension from path

    >>> extension('./file.ext')
    'ext'
    """
    basename = os.path.basename(path)
    idx = basename.find(".")
    if idx == -1:
        return None
    return basename[idx + 1 :]
