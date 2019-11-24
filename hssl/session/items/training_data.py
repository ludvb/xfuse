from typing import List

from .. import SessionItem, register_session_item


class TrainingData:
    r"""Data structure for holding training data"""
    epoch: int = 0
    step: int = 0
    elbos: List[float] = []


register_session_item(
    "training_data", SessionItem(setter=lambda _: None, default=TrainingData())
)
