import itertools as it
import os

from ..session import Session, get
from ..session.items.work_dir import WorkDir


def chdir(dirname: str) -> Session:
    r"""
    Changes the session working directory to `dirname`. Absolute paths are
    rerooted with the session root.
    """
    cwd = get("work_dir")
    if os.path.isabs(dirname):
        _root, *subdirs = os.path.normpath(dirname).split(os.sep)
        subpath = os.path.join(*subdirs)
    else:
        subpath = os.path.join(cwd.subpath, dirname)
    return Session(work_dir=WorkDir(root=cwd.root, subpath=subpath))


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
