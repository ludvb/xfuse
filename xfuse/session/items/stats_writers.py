from typing import List

from ...messengers.stats.writer import StatsWriter
from .. import SessionItem, register_session_item


__all__: List[str] = []


def _stats_writer_setter(stats_writers: List[StatsWriter]) -> None:
    # pylint: disable=unused-argument
    pass


register_session_item(
    "stats_writers",
    SessionItem(setter=_stats_writer_setter, default=[], persistent=False),
)
