from typing import Any, Dict, NamedTuple, Union, Tuple

from _io import BufferedReader

import torch as t

from ..logging import INFO, log
from ..network import Histonet, STD
from ..optimizer import create_optimizer


__all__ = [
    'State',
    'load_state',
    'save_state',
    'to_device',
]


class State(NamedTuple):
    histonet: Histonet
    std: STD
    optimizer: t.optim.Optimizer
    epoch: int


class ModuleState(NamedTuple):
    state_dict: Dict[str, t.nn.Parameter]
    init_args: Dict[str, Union[Tuple, Dict[str, Any]]]


class OptimizerState(NamedTuple):
    state_dict: Dict[str, t.nn.Parameter]


class SavedState(NamedTuple):
    histonet: ModuleState
    std: ModuleState
    optimizer: OptimizerState
    epoch: int


def save_state(state: State, file) -> None:
    if not isinstance(state, State):
        raise ValueError(f'invalid type of state {type(state)}')

    log(INFO, 'saving state to %s', file)

    t.save(
        SavedState(
            histonet=ModuleState(
                state_dict=state.histonet.state_dict(),
                init_args=state.histonet.init_args,
            ),
            std=ModuleState(
                state_dict=state.std.state_dict(),
                init_args=state.std.init_args,
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

    histonet = _reinstatiate_module(Histonet, state.histonet)
    std = _reinstatiate_module(STD, state.std)

    optimizer = create_optimizer(histonet, std)
    optimizer.load_state_dict(state.optimizer.state_dict)

    return State(
        histonet=histonet,
        std=std,
        optimizer=optimizer,
        epoch=state.epoch,
    )


def to_device(state: State, device: t.device) -> State:
    state.histonet.to(device)
    state.std.to(device)
    for ps in state.optimizer.state.values():
        p: t.Tensor
        for p in filter(t.is_tensor, ps.values()):
            p.data = p.to(device)
    return state
