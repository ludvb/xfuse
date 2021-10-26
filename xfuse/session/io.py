import pickle
import warnings
from typing import Union

import torch

from _io import BufferedReader

from . import Session, get_session
from .session import _SESSION_STORE
from ..logging import INFO, log
from ..utility.file import first_unique_filename
from ..utility.state.state import get_state_dict, load_state_dict

__all__ = ["load_session", "save_session"]


def save_session(filename_prefix: str) -> None:
    r"""Saves the current :class:`Session`"""

    def _can_pickle(name, x):
        try:
            _ = pickle.dumps(x)
        except pickle.PickleError as exc:
            warnings.warn(
                f'Session item "{name}" cannot be saved.'
                f" The error returned was: {str(exc)}",
            )
            return False
        return True

    session = Session(
        **{
            k: v
            for k, v in iter(get_session())
            if _SESSION_STORE[k].persistent
            if v is not None
            if _can_pickle(k, v)
        }
    )

    path = first_unique_filename(f"{filename_prefix}.session")
    log(INFO, "Saving session to %s", path)
    torch.save((session, get_state_dict()), path)


def load_session(file: Union[str, BufferedReader]) -> Session:
    r"""Loads :class:`Session` from a file"""
    log(
        INFO,
        "Loading session from %s",
        file.name if isinstance(file, BufferedReader) else file,
    )
    session, state_dict = torch.load(file, map_location="cpu")
    with session:
        load_state_dict(state_dict)
    return session
