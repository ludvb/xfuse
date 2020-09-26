import logging
import os
from _io import TextIOWrapper
from typing import List, Optional, Union

from ...logging import ERROR, LOGGER, log
from ...logging.formatter import Formatter
from .. import SessionItem, Unset, register_session_item


def _setter(filebuffers: Union[List[TextIOWrapper], Unset]):
    warnings_logger = logging.getLogger("py.warnings")
    while warnings_logger.handlers != []:
        warnings_logger.removeHandler(warnings_logger.handlers[0])

    while LOGGER.handlers != []:
        LOGGER.removeHandler(LOGGER.handlers[0])

    if isinstance(filebuffers, List):
        for filebuffer in filebuffers:
            fancy_formatting = filebuffer.isatty()

            handler = logging.StreamHandler(filebuffer)
            handler.setFormatter(Formatter(fancy_formatting=fancy_formatting))

            LOGGER.addHandler(handler)
            logging.getLogger("py.warnings").addHandler(handler)


register_session_item(
    "log_file", SessionItem(setter=_setter, default=[], persistent=False),
)
