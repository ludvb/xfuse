import itertools as it
import os
from typing import Optional


def first_unique_filename(root_name: str) -> str:
    r"""
    Returns the first non-existent filename in the sequence "`root_name`",
    "`root_name`.1", "`root_name`.2", ...
    """
    for path in it.chain(
        (root_name,), (f"{root_name}.{i}" for i in it.count(1))
    ):
        if not os.path.exists(path):
            return path
    raise RuntimeError("Unreachable code path")


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
