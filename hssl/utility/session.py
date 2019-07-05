import os

import pickle

from typing import Union

from _io import BufferedReader

from .file import unique_prefix
from ..logging import INFO, WARNING, log
from ..session import (
    Session,
    get_save_path,
    get_session,
)

import torch as t


__all__ = [
    'load_session',
    'save_session',
]


def save_session(filename_prefix: str) -> None:
    path = unique_prefix(os.path.join(get_save_path(), filename_prefix))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _can_pickle(name, x):
        try:
            pickle.dumps(x)
        except Exception as err:
            log(
                WARNING,
                'can\'t store session item "%s". the error returned was: %s',
                name,
                str(err),
            )
            return False
        return True

    session = Session(**{
        k: v for k, v in iter(get_session()) if _can_pickle(k, v)})

    log(INFO, 'saving session to %s', path)
    t.save(session, path)


def load_session(file: Union[str, BufferedReader]) -> Session:
    log(INFO, 'loading session from %s', str(file))
    return t.load(file)
