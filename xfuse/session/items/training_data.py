from typing import Optional

from .. import SessionItem, register_session_item


class TrainingData:
    r"""Data structure for holding training data"""
    epoch: int = 0
    step: int = 0
    elbo_long: Optional[float] = None
    elbo_short: Optional[float] = None
    rmse: Optional[float] = None


register_session_item(
    "training_data",
    SessionItem(
        setter=lambda _: None, default=TrainingData(), persistent=True
    ),
)
