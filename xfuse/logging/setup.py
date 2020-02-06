import logging

from . import DEBUG, INFO, WARNING, ERROR, LOGGER
from ..session import get


class Formatter(logging.Formatter):
    r"""Custom log message formatter"""

    def __init__(self, *args, fancy_formatting=False, **kwargs):
        self.fancy = fancy_formatting
        super().__init__(*args, **kwargs)

    def format(self, record):
        if self.fancy:
            if record.levelno >= ERROR:
                style = "\033[1m\033[91m"
            elif record.levelno >= WARNING:
                style = "\033[1m\033[93m"
            elif record.levelno >= INFO:
                style = "\033[1m"
            else:
                style = ""
            reset_style = "\033[0m"
        else:
            style = ""
            reset_style = ""
        if record.levelno == DEBUG or get("log_level") <= DEBUG:
            where = f"({record.filename}:{record.lineno})"
        else:
            where = None
        return " ".join(
            x
            for x in [
                f"[{self.formatTime(record)}]",
                "".join([style, record.levelname, reset_style]),
                where,
                ":",
                record.getMessage(),
            ]
            if x
        )


def setup_logging(filebuffer=None, fancy_formatting=None):
    r"""Adds a new logging stream"""
    if fancy_formatting is None:
        fancy_formatting = filebuffer.isatty()
    if fancy_formatting:
        logging.addLevelName(DEBUG, "🐛")
        logging.addLevelName(INFO, "ℹ")
        logging.addLevelName(WARNING, "⚠ WARNING")
        logging.addLevelName(ERROR, "💔 ERROR")
    handler = logging.StreamHandler(filebuffer)
    handler.setFormatter(Formatter(fancy_formatting=fancy_formatting))
    LOGGER.addHandler(handler)
    return handler
