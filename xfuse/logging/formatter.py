import logging

from . import DEBUG, INFO, WARNING, ERROR


LEVEL_NAMES = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
}
LEVEL_NAMES_FANCY = {
    DEBUG: "🐛",
    INFO: "ℹ",
    WARNING: "⚠ WARNING",
    ERROR: "🚨 ERROR",
}


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

        try:
            levelname = (LEVEL_NAMES_FANCY if self.fancy else LEVEL_NAMES)[
                record.levelno
            ]
        except KeyError:
            levelname = str(record.levelno)

        if record.levelno == DEBUG:
            where = f"({record.filename}:{record.lineno})"
        else:
            where = None

        return " ".join(
            x
            for x in [
                f"[{self.formatTime(record)}]",
                "".join([style, levelname, reset_style]),
                where,
                ":",
                record.getMessage(),
            ]
            if x
        )
