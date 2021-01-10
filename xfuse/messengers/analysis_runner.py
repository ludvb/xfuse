import warnings
from typing import Any, Dict, Tuple

from pyro.poutine.messenger import Messenger

from ..analyze import analyses as _analyses
from ..logging import INFO, log
from ..session import Session, get
from ..utility.file import chdir


class AnalysisRunner(Messenger):
    r"""Saves the currently running session to disk at a fixed interval"""

    def __init__(
        self,
        analyses: Dict[str, Tuple[str, Dict[str, Any]]],
        period: int = 10000,
    ):
        super().__init__()
        self._analyses = analyses
        self._period = period

    def _pyro_post_step(self, _msg):
        if (step := get("training_data").step) % self._period == 0:
            for name, (analysis_type, options) in self._analyses.items():
                if analysis_type in _analyses:
                    log(INFO, 'Running analysis "%s"', name)
                    with Session(messengers=[]):
                        with chdir(f"/analyses/step-{step:06d}/{name}"):
                            _analyses[analysis_type].function(**options)
                else:
                    warnings.warn(f'Unknown analysis "{analysis_type}"')
