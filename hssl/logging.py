import logging
from logging import ERROR, WARNING, INFO, DEBUG

import sys


class Formatter(logging.Formatter):
    """ Custom log message formatter
    """

    def __init__(self, *args, fancy_formatting=False, **kwargs):
        self.fancy = fancy_formatting
        super().__init__(*args, **kwargs)

    def format(self, record):
        return ''.join(filter(lambda x: x is not None, (
            # pylint: disable=line-too-long
            f'[{self.formatTime(record)}]',
            ' ',
            '\033[1m'  if self.fancy and record.levelno >= INFO                               else None,
            '\033[91m' if self.fancy and record.levelno >= ERROR                              else None,
            '\033[93m' if self.fancy and record.levelno >= WARNING and record.levelno < ERROR else None,
            record.levelname.lower(),
            '\033[0m'  if self.fancy else None,
            (
                f' ({record.filename}:{record.lineno})'
                if record.levelno != INFO else None
            ),
            ': ',
            record.getMessage(),
        )))


HANDLER = logging.StreamHandler(sys.stderr)
HANDLER.setFormatter(Formatter(fancy_formatting=sys.stderr.isatty()))

logging.basicConfig(
    handlers=[HANDLER],
)

LOGGER = logging.getLogger(__name__)

log = LOGGER.log


def set_level(level: int):
    """
    Parameters
    ----------
    level : int
        log level

    Side effects
    ------------
    Sets the log level to `level`

    Returns
    -------
    None
    """
    LOGGER.setLevel(level)
