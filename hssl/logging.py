from contextlib import ContextDecorator

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


class LoggedExecution(ContextDecorator):
    def __init__(self, log_file=None):
        self.log_file = log_file
        self._file_handle = None
        self._log_handler = None

    def __enter__(self):
        if self.log_file is not None:
            self._file_handle = open(self.log_file, 'a')
            from logging import StreamHandler
            self._log_handler = StreamHandler(self._file_handle)
            self._log_handler.setFormatter(Formatter(fancy_formatting=False))
            LOGGER.addHandler(self._log_handler)

    def __exit__(self, err_type, err, tb):
        if err is not None:
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            LOGGER.findCaller = (
                lambda self, stack_info=None, f=frame:
                (f.f_code.co_filename, f.f_lineno, f.f_code.co_name, None)
            )
            log(ERROR, str(err))
        if self._log_handler is not None:
            LOGGER.removeHandler(self._log_handler)
            self._file_handle.close()
