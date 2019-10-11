import logging
import sys
from functools import wraps
from logging import DEBUG, ERROR, INFO, WARNING


class Formatter(logging.Formatter):
    """ Custom log message formatter
    """

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
        if record.levelno in [DEBUG, WARNING, ERROR]:
            where = f"({record.filename}:{record.lineno})"
        else:
            where = None
        return " ".join(
            x
            for x in [
                f"[{self.formatTime(record)}]",
                "".join([style, record.levelname.lower(), reset_style]),
                where,
                ":",
                record.getMessage(),
            ]
            if x
        )


HANDLER = logging.StreamHandler(sys.stderr)
HANDLER.setFormatter(Formatter(fancy_formatting=sys.stderr.isatty()))

logging.basicConfig(handlers=[HANDLER])

LOGGER = logging.getLogger(__name__)


@wraps(LOGGER.log)
def log(*args, **kwargs):
    # pylint: disable=missing-function-docstring
    return LOGGER.log(*args, **kwargs)


def set_level(level: int):
    """Set logging level """
    LOGGER.setLevel(level)
