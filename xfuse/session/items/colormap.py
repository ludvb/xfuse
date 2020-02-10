from matplotlib.cm import inferno  # pylint: disable=no-name-in-module
from .. import SessionItem, register_session_item


register_session_item(
    "colormap",
    SessionItem(setter=lambda _: None, default=inferno, persistent=False),
)
