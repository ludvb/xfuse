from typing import Any, Dict, NamedTuple, Union, Tuple

from _io import BufferedReader

import torch as t

from ..logging import INFO, log
from ..network import XFuse
from ..optimizer import create_optimizer


__all__ = [
    'State',
    'load_state',
    'save_state',
    'to_device',
]


class State(NamedTuple):
    model: XFuse
    optimizer: t.optim.Optimizer
    epoch: int


class ModuleState(NamedTuple):
    state_dict: Dict[str, t.nn.Parameter]
    init_args: Dict[str, Union[Tuple, Dict[str, Any]]]


class OptimizerState(NamedTuple):
    state_dict: Dict[str, t.nn.Parameter]


class SavedState(NamedTuple):
    model: ModuleState
    optimizer: OptimizerState
    epoch: int


def save_state(state: State, file) -> None:
    if not isinstance(state, State):
        raise ValueError(f'invalid type of state {type(state)}')

    log(INFO, 'saving state to %s', file)

    t.save(
        SavedState(
            model=ModuleState(
                state_dict=state.model.state_dict(),
                init_args=state.model.init_args,
            ),
            optimizer=OptimizerState(
                state_dict=state.optimizer.state_dict(),
            ),
            epoch=state.epoch,
        ),
        file,
    )


def load_state(file: Union[str, BufferedReader]) -> State:
    log(INFO, 'loading state from %s',
        file.name if isinstance(file, BufferedReader) else file)

    state: SavedState = t.load(file, map_location=t.device('cpu'))
    if not isinstance(state, SavedState):
        raise ValueError(f'invalid type of state {type(state)}')

    def _reinstatiate_module(module, state):
        instance = module(
            *state.init_args['args'],
            **state.init_args['kwargs'],
        )
        instance.load_state_dict(state.state_dict)
        return instance

    model = _reinstatiate_module(XFuse, state.model)

    optimizer = create_optimizer(model)
    optimizer.load_state_dict(state.optimizer.state_dict)

    return State(
        model=model,
        optimizer=optimizer,
        epoch=state.epoch,
    )


def to_device(state: State, device: t.device) -> State:
    state.model.to(device)
    for ps in state.optimizer.state.values():
        p: t.Tensor
        for p in filter(t.is_tensor, ps.values()):
            p.data = p.to(device)
    return state
