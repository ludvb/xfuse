from .. import SessionItem, register_session_item


register_session_item(
    "covariates",
    SessionItem(setter=lambda _: None, default={}, persistent=True),
)
