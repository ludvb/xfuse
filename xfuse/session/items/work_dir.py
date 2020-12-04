import os
from typing import List, Optional

from .. import SessionItem, register_session_item
from ...logging import DEBUG, log


__all__: List[str] = []


class WorkDir:
    """Stores the current working directory"""

    def __init__(self, root: Optional[str] = None, subpath: str = os.curdir):
        if root is None:
            root = os.getcwd()
        self.root = root
        self.subpath = subpath

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkDir):
            raise NotImplementedError()
        return other.full_path == self.full_path

    @property
    def root(self) -> str:
        """The root file path"""
        return self.__root

    @root.setter
    def root(self, root: str):
        """Setter for the root file path"""
        root = os.path.expanduser(root)
        root = os.path.expandvars(root)
        root = os.path.abspath(root)
        root = os.path.normcase(root)
        root = os.path.normpath(root)
        self.__root = root

    @property
    def subpath(self) -> str:
        """The currently active subdirectory"""
        return self.__subpath

    @subpath.setter
    def subpath(self, subpath: str):
        """Setter for the currently active subdirectory"""
        subpath = os.path.expandvars(subpath)
        subpath = os.path.normcase(subpath)
        subpath = os.path.normpath(subpath)
        self.__subpath = subpath

    @property
    def full_path(self) -> str:
        """The full path (:func:`root` + :func:`subpath`)"""
        return os.path.join(self.root, self.subpath)


__DEFAULT_WORKDIR = WorkDir()
__CUR_WORKDIR = __DEFAULT_WORKDIR


def _work_dir_setter(work_dir: WorkDir) -> None:
    # pylint: disable=global-statement
    global __CUR_WORKDIR
    if work_dir != __CUR_WORKDIR:
        log(DEBUG, "Changing working directory to: %s", work_dir.full_path)
        if not os.path.exists(work_dir.full_path):
            os.makedirs(work_dir.full_path, exist_ok=True)
        os.chdir(work_dir.full_path)
        __CUR_WORKDIR = work_dir


register_session_item(
    "work_dir",
    SessionItem(setter=_work_dir_setter, default=WorkDir(), persistent=False),
)
