from abc import abstractmethod

import matplotlib.pyplot as plt
import pyro.poutine
import torch

from ...analyze.metagenes import (
    compute_metagene_profiles,
    visualize_metagene_profile,
)
from ...model.experiment.st import ST
from ...session import get
from ...utility.visualization import reduce_last_dimension, visualize_metagenes
from .stats_handler import (
    StatsHandler,
    log_figure,
    log_histogram,
    log_image,
    log_images,
    log_scalar,
)

__all__ = [
    "MetageneHistogram",
    "MetageneMean",
    "MetageneSummary",
    "MetageneFullSummary",
]


class Metagene(StatsHandler):
    r"""Abstract class for metagene trackers"""

    def _select_msg(self, type, name, **_):
        # pylint: disable=arguments-differ
        # pylint: disable=redefined-builtin
        return type == "sample" and name[-3:] == "rim"

    @abstractmethod
    def _handle_metagene(self, name, metagene):
        pass

    def _handle(self, fn, **_):
        # pylint: disable=arguments-differ
        if (model := get("model")) is not None:
            experiment: ST = model.get_experiment("ST")
            for name, metagene in zip(
                experiment.metagenes, fn.mean.permute(1, 0, 2, 3)
            ):
                self._handle_metagene(name, metagene)


class MetageneHistogram(Metagene):
    r"""Summarizes the spatial activation of each metagene in a histogram"""

    def _handle_metagene(self, name, metagene):
        # pylint: disable=no-member
        log_histogram(
            f"metagene-histogram/metagene-{name}", metagene.flatten()
        )


class MetageneMean(Metagene):
    r"""Summarizes the mean spatial activation of each metagene"""

    def _handle_metagene(self, name, metagene):
        # pylint: disable=no-member
        log_scalar(f"metagene-mean/metagene-{name}", metagene.mean())


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
            log_images(
                "metagene-batch-summary",
                torch.as_tensor(
                    reduce_last_dimension(fn.mean.permute(0, 2, 3, 1))
                ),
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
                for (
                    slide_name,
                    summarization,
                    metagenes,
                ) in visualize_metagenes():
                    # pylint: disable=no-member
                    log_image(
                        f"metagene-summary/{slide_name}",
                        torch.as_tensor(summarization),
                    )
                    for name, metagene in metagenes:
                        log_image(
                            f"metagene-{name}/{slide_name}",
                            torch.as_tensor(metagene),
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
                log_figure(
                    f"metagene-{name}/profile/{experiment}/invcvsort", fig,
                )

                fig = plt.figure(figsize=(3.5, 3.7))
                visualize_metagene_profile(
                    profile.loc[name], num_high=20, num_low=10, sort_by="mean",
                )
                plt.tight_layout(pad=0.0)
                log_figure(
                    f"metagene-{name}/profile/{experiment}/meansort", fig,
                )
