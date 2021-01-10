import random
from typing import Optional

import torch
import numpy as np

from .. import SessionItem, register_session_item


__all__ = ["RNGState"]


class RNGState:
    def __init__(self, seed: Optional[int] = None):
        # Temporarily seed RNGs to get initial states and then restore them
        current_random_state = random.getstate()
        current_numpy_state = np.random.get_state()
        current_torch_state = torch.random.get_rng_state()
        random.seed(seed)
        np.random.seed(random.randint(0, np.iinfo(np.int32).max))
        torch.manual_seed(random.randint(0, np.iinfo(np.int32).max))
        self._update_state()
        random.setstate(current_random_state)
        np.random.set_state(current_numpy_state)
        torch.random.set_rng_state(current_torch_state)

    def _update_state(self):
        self.__random_state = random.getstate()
        self.__numpy_state = np.random.get_state()
        self.__torch_state = torch.random.get_rng_state()

    def activate(self):
        random.setstate(self.__random_state)
        np.random.set_state(self.__numpy_state)
        torch.random.set_rng_state(self.__torch_state)


_DEFAULT_RNG_STATE = RNGState()
_CURRENT_RNG_STATE = _DEFAULT_RNG_STATE


def _rng_state_setter(state: RNGState) -> None:
    global _CURRENT_RNG_STATE
    _CURRENT_RNG_STATE._update_state()
    _CURRENT_RNG_STATE = state
    _CURRENT_RNG_STATE.activate()


register_session_item(
    "rng_state",
    SessionItem(
        setter=_rng_state_setter, default=_DEFAULT_RNG_STATE, persistent=False,
    ),
)
