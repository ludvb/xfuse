from abc import abstractmethod

import numpy as np
import pyro.poutine
import torch

from ...data import Data, Dataset
from ...data.slide import FullSlide, Slide
from ...data.utility.misc import make_dataloader
from ...logging import WARNING, log
from ...model.experiment.st import ST
from ...session import Session, require
from ...utility.visualization import reduce_last_dimension
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

    def _select_msg(self, type, name, value, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name[-3:] == "rim" and value.shape[1] >= 3

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=no-member
        self.add_images(
            "activation-summary/training-batch",
            reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
            dataformats="NHWC",
        )


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
        model = require("model")
        dataloader = require("dataloader")

        def _compute_factor_activation(xs):
            if len(xs) != 1:
                raise RuntimeError()
            if "ST" not in xs:
                raise NotImplementedError()
            data = xs["ST"]
            with torch.no_grad():
                with pyro.poutine.trace() as guide_trace:
                    model.guide(xs)
                with pyro.poutine.replay(trace=guide_trace.trace):
                    with pyro.poutine.trace() as model_trace:
                        model.model(xs)
            zero_label = torch.where(data["data"][0].sum(1) == 0)[0] + 1
            mask = ~np.isin(data["label"][0], zero_label)
            return (
                model_trace.trace.nodes["rim"]["fn"].mean[0].permute(1, 2, 0),
                mask,
            )

        dataloader = make_dataloader(
            Dataset(
                Data(
                    slides={
                        k: Slide(data=v.data, iterator=FullSlide)
                        for k, v in dataloader.dataset.data.slides.items()
                    },
                    design=dataloader.dataset.data.design,
                )
            ),
            batch_size=1,
            shuffle=False,
        )

        with Session(default_device=torch.device("cpu"), pyro_stack=[]):
            for i, x in enumerate(dataloader, 1):
                factor_activation, mask = _compute_factor_activation(x)
                n_components = 3 if factor_activation.shape[-1] >= 3 else 1
                reduced_factor_activation = reduce_last_dimension(
                    factor_activation, mask=mask, n_components=n_components
                )
                # pylint: disable=no-member
                self.add_image(
                    f"activation-summary/sample{i}",
                    torch.as_tensor(reduced_factor_activation),
                    dataformats="HWC",
                )
