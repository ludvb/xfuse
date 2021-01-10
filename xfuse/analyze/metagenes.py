import os
import warnings
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import pyro
from imageio import imwrite

from ..model.experiment.st.st import ST, _encode_metagene_name
from ..session import Session, require
from ..utility.visualization import visualize_metagenes
from .analyze import Analysis, _register_analysis

__all__ = [
    "compute_metagene_profiles",
    "compute_metagene_summary",
    "visualize_metagene_profile",
]


def compute_metagene_profiles():
    r"""Computes metagene profiles"""
    model = require("model")
    genes = require("genes")

    def _metagene_profile_st():
        model = require("model")
        experiment = cast(ST, model.get_experiment("ST"))
        with pyro.poutine.block():
            with pyro.poutine.trace() as trace:
                # pylint: disable=protected-access
                experiment._sample_metagenes()
        return [
            (n, trace.trace.nodes[_encode_metagene_name(n)]["fn"])
            for n in experiment.metagenes
        ]

    _metagene_profile_fn = {"ST": _metagene_profile_st}

    for experiment in model.experiments.keys():
        try:
            fn = _metagene_profile_fn[experiment]
        except KeyError:
            warnings.warn(
                f'Metagene profiles for experiment of type "{experiment}"'
                " not implemented"
            )
            continue

        names, profiles = zip(*fn())
        dataframe = (
            pd.concat(
                [
                    pd.DataFrame(
                        [
                            x.mean.detach().cpu().numpy(),
                            x.stddev.detach().cpu().numpy(),
                        ],
                        columns=genes,
                        index=pd.Index(["mean", "stddev"], name="log2fold"),
                    )
                    for x in profiles
                ],
                keys=pd.Index(names, name="metagene"),
            )
            .reset_index()
            .melt(
                ["metagene", "log2fold"], var_name="gene", value_name="value",
            )
            .pivot_table(
                index=["metagene", "gene"], columns="log2fold", values="value",
            )
        )
        yield experiment, dataframe


def visualize_metagene_profile(
    profile, num_high=20, num_low=20, sort_by="mean", ax=None,
):
    r"""Creates metagene profile visualization"""
    num_low = max(min(len(profile) - num_high, num_low), 0)
    x = profile.sort_values(sort_by)
    x = pd.concat([x.iloc[:num_low], x.iloc[-num_high:]])
    (ax if ax else plt).errorbar(
        x["mean"], x.index, xerr=x["stddev"], fmt="none", c="black"
    )
    (ax if ax else plt).vlines(
        0.0,
        ymin=x.index[0],
        ymax=x.index[-1],
        colors="red",
        linestyles="--",
        lw=1,
    )


def compute_metagene_summary(method: str = "pca") -> None:
    r"""Computes metagene summary"""
    # pylint: disable=too-many-locals
    with Session(messengers=[]):
        for (slide_name, summarization, metagenes) in visualize_metagenes(
            method
        ):
            os.makedirs(slide_name, exist_ok=True)
            imwrite(
                os.path.join(slide_name, "summary.png"), summarization,
            )
            for name, metagene in metagenes:
                imwrite(
                    os.path.join(slide_name, f"metagene-{name}.png"), metagene,
                )

        for experiment, metagene_profiles in compute_metagene_profiles():
            metagene_profiles.to_csv(f"{experiment}-metagene-log2fold.csv.gz")
            metagene_profiles["invcv"] = (
                metagene_profiles["mean"] / metagene_profiles["stddev"]
            )
            for metagene, profile in metagene_profiles.groupby(level=0):
                plt.figure(figsize=(4, 10))
                visualize_metagene_profile(
                    profile.loc[metagene],
                    num_high=30,
                    num_low=15,
                    sort_by="invcv",
                )
                plt.title(f"{metagene=} ({experiment})")
                plt.tight_layout(pad=0.0)
                plt.savefig(
                    f"{experiment}-metagene-{metagene}-invcvsort.png", dpi=600,
                )
                plt.close()

                plt.figure(figsize=(4, 10))
                visualize_metagene_profile(
                    profile.loc[metagene],
                    num_high=30,
                    num_low=15,
                    sort_by="mean",
                )
                plt.title(f"{metagene=} ({experiment})")
                plt.tight_layout(pad=0.0)
                plt.savefig(
                    f"{experiment}-metagene-{metagene}-meansort.png", dpi=600,
                )
                plt.close()


_register_analysis(
    name="metagenes",
    analysis=Analysis(
        description="Creates summary data of the metagenes",
        function=compute_metagene_summary,
    ),
)
