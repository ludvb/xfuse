import os
import pickle
from typing import Union

import torch as t

from _io import BufferedReader

from ..logging import INFO, WARNING, log
from ..session import Session, get_session, require
from .file import first_unique_filename
from .modules import get_state_dict, load_state_dict

__all__ = ["load_session", "save_session"]


def save_session(filename_prefix: str) -> None:
    r"""Saves the current :class:`Session`"""
    try:
        save_path = require("save_path")
    except RuntimeError:
        return

    path = first_unique_filename(
        os.path.join(save_path, f"{filename_prefix}.session")
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _can_pickle(name, x):
        try:
            pickle.dumps(x)
        except Exception as err:  # pylint: disable=broad-except
            log(
                WARNING,
                'can\'t store session item "%s". the error returned was: %s',
                name,
                str(err),
            )
            return False
        return True

    session = Session(
        **{k: v for k, v in iter(get_session()) if _can_pickle(k, v)}
    )

    log(INFO, "saving session to %s", path)
    t.save((session, get_state_dict()), path)


def load_session(file: Union[str, BufferedReader]) -> Session:
    r"""Loads :class:`Session` from a file"""
    log(INFO, "loading session from %s", str(file))
    session, state_dict = t.load(file)
    load_state_dict(state_dict)
    return session
