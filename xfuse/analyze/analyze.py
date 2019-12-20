from typing import Callable, Dict, NamedTuple

from ..logging import DEBUG, log


class Analysis(NamedTuple):
    r"""Data type for analyses"""
    description: str
    function: Callable[..., None]


_ANALYSES: Dict[str, Analysis] = {}


def _register_analysis(name, analysis: Analysis):
    if name not in _ANALYSES:
        log(DEBUG, 'Registering analysis "%s"', name)
        _ANALYSES[name] = analysis
    else:
        raise RuntimeError(f'Analysis "{name}" has already been registered!')
