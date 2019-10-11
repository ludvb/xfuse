from logging import StreamHandler

import os

from typing import Optional

from .session_item import SessionItem
from ...logging import DEBUG, Formatter, LOGGER, log


_LOG_HANDLER = None
_FILE_STREAM = None


def _setter(path: Optional[str]):
    global _FILE_STREAM, _LOG_HANDLER
    if _LOG_HANDLER is not None:
        LOGGER.removeHandler(_LOG_HANDLER)
        _LOG_HANDLER = None
    if _FILE_STREAM is not None:
        _FILE_STREAM.close()
        _FILE_STREAM = None
    if path is not None:
        log(DEBUG, "opening log file stream: %s", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _FILE_STREAM = open(path, "a")
        _LOG_HANDLER = StreamHandler(_FILE_STREAM)
        _LOG_HANDLER.setFormatter(Formatter(fancy_formatting=False))
        LOGGER.addHandler(_LOG_HANDLER)


log_file = SessionItem(setter=_setter, default=None)
