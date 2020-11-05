import matplotlib
from .. import SessionItem, register_session_item


register_session_item(
    "mpl_backend",
    SessionItem(
        setter=matplotlib.use,
        default=matplotlib.get_backend(),
        persistent=False,
    ),
)
