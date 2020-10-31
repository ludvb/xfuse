import warnings
from abc import abstractmethod

import matplotlib.pyplot as plt
import pyro.poutine
import torch

from ...analyze.metagenes import (
    compute_metagene_profiles,
    visualize_metagene_profile,
)
from ...model.experiment.st import ST
from ...session import require
from ...utility.visualization import reduce_last_dimension, visualize_metagenes
from .. import Noop
from .stats_handler import StatsHandler

__all__ = [
    "MetageneHistogram",
    "MetageneMean",
    "MetageneSummary",
    "MetageneFullSummary",
]


class Metagene(StatsHandler):
    r"""Abstract class for metagene trackers"""

    _experiment: ST

    def __new__(cls, *_args, **_kwargs):
        try:
            st_experiment: ST = require("model").get_experiment("ST")
        except (AttributeError, KeyError):
            warnings.warn(
                "Session model does not have an ST experiment."
                f" {cls.__name__} will be disabled."
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
    def _handle_metagene(self, name, metagene):
        pass

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        for name, metagene in zip(
            self._experiment.metagenes, fn.mean.permute(1, 0, 2, 3)
        ):
            self._handle_metagene(name, metagene)


class MetageneHistogram(Metagene):
    r"""Summarizes the spatial activation of each metagene in a histogram"""

    def _handle_metagene(self, name, metagene):
        # pylint: disable=no-member
        self.add_histogram(
            f"metagene-histogram/metagene-{name}", metagene.flatten()
        )


class MetageneMean(Metagene):
    r"""Summarizes the mean spatial activation of each metagene"""

    def _handle_metagene(self, name, metagene):
        # pylint: disable=no-member
        self.add_scalar(f"metagene-mean/metagene-{name}", metagene.mean())


class MetageneSummary(StatsHandler):
    r"""Plots summarized spatial activations of all metagenes"""

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
                "metagene-batch-summary",
                reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
                dataformats="NHWC",
            )
        except ValueError:
            pass


class MetageneFullSummary(StatsHandler):
    r"""Plots summarized spatial activations of all metagenes in each sample"""

    def _select_msg(self, type, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "step"

    def _handle(self, **msg):
        try:
            with pyro.poutine.block():
                for i, (summarization, metagenes) in enumerate(
                    visualize_metagenes(), 1,
                ):
                    # pylint: disable=no-member
                    self.add_image(
                        f"metagene-summary/sample-{i}",
                        torch.as_tensor(summarization),
                        dataformats="HWC",
                    )
                    for name, metagene in metagenes:
                        self.add_image(
                            f"metagene-{name}/sample-{i}",
                            torch.as_tensor(metagene),
                            dataformats="HWC",
                        )
        except ValueError:
            pass

        for experiment, metagene_profiles in compute_metagene_profiles():
            metagene_profiles["invcv"] = (
                metagene_profiles["mean"] / metagene_profiles["stddev"]
            )
            for name, profile in metagene_profiles.groupby(level=0):
                fig = plt.figure(figsize=(3.5, 3.7))
                visualize_metagene_profile(
                    profile.loc[name],
                    num_high=20,
                    num_low=10,
                    sort_by="invcv",
                )
                plt.tight_layout(pad=0.0)
                # pylint: disable=no-member
                self.add_figure(
                    f"metagene-{name}/profile/{experiment}/invcvsort", fig,
                )

                fig = plt.figure(figsize=(3.5, 3.7))
                visualize_metagene_profile(
                    profile.loc[name], num_high=20, num_low=10, sort_by="mean",
                )
                plt.tight_layout(pad=0.0)
                # pylint: disable=no-member
                self.add_figure(
                    f"metagene-{name}/profile/{experiment}/meansort", fig,
                )
