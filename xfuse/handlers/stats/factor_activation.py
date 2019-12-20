from abc import abstractmethod

import pyro.poutine
import torch

from ...logging import WARNING, log
from ...model.experiment.st import ST
from ...session import require
from ...utility.visualization import reduce_last_dimension, visualize_factors
from .. import Noop
from .stats_handler import StatsHandler

__all__ = [
    "FactorActivationHistogram",
    "FactorActivationMaps",
    "FactorActivationMean",
    "FactorActivationSummary",
    "FactorActivationFullSummary",
]


class FactorActivation(StatsHandler):
    r"""Abstract class for factor activation trackers"""

    _experiment: ST

    def __new__(cls, *_args, **_kwargs):
        try:
            st_experiment: ST = require("model").get_experiment("ST")
        except (AttributeError, KeyError):
            log(
                WARNING,
                "session model does not have an ST experiment."
                f" {cls.__name__} will be disabled.",
            )
            return Noop()
        instance = super().__new__(cls)
        instance._experiment = st_experiment
        return instance

    def _select_msg(self, type, name, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name[-3:] == "rim"

    @abstractmethod
    def _handle_factor_activation(self, name, activation):
        pass

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        for name, factor in zip(
            self._experiment.factors, fn.mean.permute(1, 0, 2, 3)
        ):
            self._handle_factor_activation(name, factor)


class FactorActivationHistogram(FactorActivation):
    r"""
    Factor activation tracker, summarizing factor activities in a histogram
    """

    def _handle_factor_activation(self, name, activation):
        # pylint: disable=no-member
        self.add_histogram(f"activation/factor{name}", activation.flatten())


class FactorActivationMaps(FactorActivation):
    r"""
    Factor activation tracker, summarizing factor activities in spatial
    activation maps
    """

    def _handle_factor_activation(self, name, activation):
        # pylint: disable=no-member
        self.add_images(
            f"activation/maps/factor{name}",
            activation.unsqueeze(1),
            dataformats="NCHW",
        )


class FactorActivationMean(FactorActivation):
    r"""
    Factor activation tracker, summarizing factor activity means
    """

    def _handle_factor_activation(self, name, activation):
        # pylint: disable=no-member
        self.add_scalar(f"activation/mean/factor{name}", activation.mean())


class FactorActivationSummary(StatsHandler):
    r"""
    Factor activation tracker, summarizing factor activities in a spatial
    activation map
    """

    def _select_msg(self, type, name, value, **msg):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return (
            type == "sample"
            and not msg["is_guide"]
            and name[-3:] == "rim"
            and value.shape[1] >= 3
        )

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        try:
            self.add_images(
                "activation-summary/training-batch",
                reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
                dataformats="NHWC",
            )
        except ValueError:
            pass


class FactorActivationFullSummary(StatsHandler):
    r"""
    Factor activation tracker, summarizing factor activities in a spatial
    activation map
    """

    def _select_msg(self, type, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"

    def _handle(self, **msg):
        try:
            with pyro.poutine.block():
                for i, factor_activation in enumerate(visualize_factors()):
                    # pylint: disable=no-member
                    self.add_image(
                        f"activation-summary/sample{i}",
                        torch.as_tensor(factor_activation),
                        dataformats="HWC",
                    )
        except ValueError:
            pass
