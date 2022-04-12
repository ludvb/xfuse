# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

import itertools as it
import json
import logging
import os
import sys
import warnings
from datetime import datetime as dt
from functools import wraps

import click
import cv2 as cv
import h5py
import numpy as np
import pandas as pd
import tomlkit

from imageio import imread
from PIL import Image
from tabulate import tabulate

from . import __version__, convert
from ._config import (  # type: ignore
    construct_default_config_toml,
    merge_config,
)
from .logging import DEBUG, INFO, log
from .messengers import AnalysisRunner, Checkpointer, stats as stats_trackers
from .messengers.stats.writer import FileWriter, TensorboardWriter
from .model.experiment.st.metagene_expansion_strategy import (
    STRATEGIES as expansion_strategies,
)
from .run import run as _run
from .utility.core import temp_attr
from .utility.file import first_unique_filename
from .session import Session, get
from .session.io import load_session
from .session.items.work_dir import WorkDir


def _init(f):
    @click.option(
        "--save-path",
        type=click.Path(),
        default=f"xfuse-{dt.now().isoformat()}",
        help="The output path",
        show_default=True,
    )
    @click.option("--debug", is_flag=True)
    @wraps(f)
    def _wrapped(*args, debug, save_path, **kwargs):
        logging.captureWarnings(True)
        with Session(work_dir=WorkDir(save_path)):
            log_filename = first_unique_filename("log")
            with open(log_filename, "w") as log_file:
                with Session(
                    log_file=[sys.stderr, log_file],
                    log_level=DEBUG if debug else INFO,
                ):
                    sys.excepthook = lambda *_: None
                    with temp_attr(
                        warnings,
                        "formatwarning",
                        lambda message, category, filename, lineno, _line: (
                            f"{category.__name__:s}"
                            f" ({filename:s}:{lineno:d}):"
                            f" {str(message):s}"
                        ),
                    ):
                        log(
                            INFO,
                            "Running %s version %s",
                            __package__,
                            __version__,
                        )
                        log(DEBUG, "Invoked by `%s`", " ".join(sys.argv))
                        return f(*args, **kwargs)

    return _wrapped


@click.group()
@click.version_option()
def cli():
    pass


@click.group("convert")
def _convert():
    r"""Converts data of various formats to the format used by xfuse."""


cli.add_command(_convert)


@click.command()
@click.option("--image", type=click.File("rb"), required=True)
@click.option("--bc-matrix", type=click.File("rb"), required=True)
@click.option("--tissue-positions", type=click.File("rb"), required=True)
@click.option("--annotation", type=click.File("rb"))
@click.option("--scale-factors", type=click.File("rb"), required=True)
@click.option("--scale", type=float)
@click.option("--mask/--no-mask", default=True)
@click.option(
    "--mask-file",
    type=click.File("rb"),
    help=" ".join(
        [
            "Custom mask.",
            "Should be a single-channel image with the same size as the image.",
            "Uses the following encoding: {}.".format(
                ", ".join(
                    sorted(
                        [
                            f"{cv.GC_BGD}=background",
                            f"{cv.GC_PR_BGD}=likely background",
                            f"{cv.GC_FGD}=foreground",
                            f"{cv.GC_PR_FGD}=likely foreground",
                        ]
                    )
                )
            ),
        ]
    ),
)
@click.option("--rotate/--no-rotate", default=True)
@_init
def _convert_visium(
    image,
    bc_matrix,
    tissue_positions,
    annotation,
    scale_factors,
    scale,
    mask,
    mask_file,
    rotate,
):
    r"""Converts 10X Visium data"""
    tissue_positions = pd.read_csv(tissue_positions, index_col=0, header=None)
    tissue_positions = tissue_positions[[1, 4, 5]]
    tissue_positions = tissue_positions.rename(
        columns={1: "in_tissue", 4: "y", 5: "x"}
    )

    scale_factors = json.load(scale_factors)
    spot_radius = scale_factors["spot_diameter_fullres"] / 2

    with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
        image_data = imread(image)

    if annotation:
        with h5py.File(annotation, "r") as annotation_file:
            annotation = {
                k: annotation_file[k][()] for k in annotation_file.keys()
            }

    if mask_file:
        with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
            custom_mask = imread(mask_file)
    else:
        custom_mask = None

    with h5py.File(bc_matrix, "r") as data:
        convert.visium.run(
            image_data,
            data,
            tissue_positions,
            spot_radius,
            output_file="data.h5",
            annotation=annotation,
            scale_factor=scale,
            mask=mask,
            custom_mask=custom_mask,
            rotate=rotate,
        )


_convert.add_command(_convert_visium, "visium")


@click.command()
@click.option("--counts", type=click.File("rb"), required=True)
@click.option("--image", type=click.File("rb"), required=True)
@click.option("--spots", type=click.File("rb"))
@click.option("--transformation-matrix", type=click.File("rb"))
@click.option("--annotation", type=click.File("rb"))
@click.option("--scale", type=float)
@click.option("--mask/--no-mask", default=True)
@click.option(
    "--mask-file",
    type=click.File("rb"),
    help=" ".join(
        [
            "Custom mask.",
            "Should be a single-channel image with the same size as the image.",
            "Uses the following encoding: {}.".format(
                ", ".join(
                    sorted(
                        [
                            f"{cv.GC_BGD}=background",
                            f"{cv.GC_PR_BGD}=likely background",
                            f"{cv.GC_FGD}=foreground",
                            f"{cv.GC_PR_FGD}=likely foreground",
                        ]
                    )
                )
            ),
        ]
    ),
)
@click.option("--rotate/--no-rotate", default=True)
@_init
def _convert_st(
    counts,
    image,
    spots,
    transformation_matrix,
    annotation,
    scale,
    mask,
    mask_file,
    rotate,
):
    r"""Converts Spatial Transcriptomics ("ST") data"""
    if spots is not None and transformation_matrix is not None:
        raise RuntimeError(
            "Please pass either a spot data file or a text file containing a"
            " transformation matrix"
        )

    if spots is not None:
        spots_data = pd.read_csv(spots, sep="\t")
    else:
        spots_data = None

    if transformation_matrix is not None:
        transformation = np.loadtxt(transformation_matrix)
        transformation = transformation.reshape(3, 3)
    else:
        transformation = None

    counts_data = pd.read_csv(counts, sep="\t", index_col=0)

    with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
        image_data = imread(image)

    if annotation:
        with h5py.File(annotation, "r") as annotation_file:
            annotation = {
                k: annotation_file[k][()] for k in annotation_file.keys()
            }

    if mask_file:
        with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
            custom_mask = imread(mask_file)
    else:
        custom_mask = None

    convert.st.run(
        counts_data,
        image_data,
        output_file="data.h5",
        spots=spots_data,
        transformation=transformation,
        annotation=annotation,
        scale_factor=scale,
        mask=mask,
        custom_mask=custom_mask,
        rotate=rotate,
    )


_convert.add_command(_convert_st, "st")


@click.command()
@click.option("--image", type=click.File("rb"), required=True)
@click.option("--annotation", type=click.File("rb"))
@click.option("--scale", type=float)
@click.option("--mask/--no-mask", default=True)
@click.option(
    "--mask-file",
    type=click.File("rb"),
    help=" ".join(
        [
            "Custom mask.",
            "Should be a single-channel image with the same size as the image.",
            "Uses the following encoding: {}.".format(
                ", ".join(
                    sorted(
                        [
                            f"{cv.GC_BGD}=background",
                            f"{cv.GC_PR_BGD}=likely background",
                            f"{cv.GC_FGD}=foreground",
                            f"{cv.GC_PR_FGD}=likely foreground",
                        ]
                    )
                )
            ),
        ]
    ),
)
@click.option("--rotate/--no-rotate", default=False)
@_init
def _convert_image(
    image, annotation, scale, mask, mask_file, rotate,
):
    r"""Converts image without any associated expression data"""
    with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
        image_data = imread(image)

    if annotation:
        with h5py.File(annotation, "r") as annotation_file:
            annotation = {
                k: annotation_file[k][()] for k in annotation_file.keys()
            }

    if mask_file:
        with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
            custom_mask = imread(mask_file)
    else:
        custom_mask = None

    convert.image.run(
        image_data,
        output_file="data.h5",
        annotation=annotation,
        scale_factor=scale,
        mask=mask,
        custom_mask=custom_mask,
        rotate=rotate,
    )


_convert.add_command(_convert_image, "image")


@click.command()
@click.argument("target", type=click.Path(), default=f"{__package__}.toml")
@click.argument(
    "slides", type=click.Path(exists=True, dir_okay=False), nargs=-1
)
def init(target, slides):
    r"""Creates a template for the project configuration file."""
    config = construct_default_config_toml()

    config["slides"].update(
        {
            f"section{i:d}": {
                "data": str(slide),
                "covariates": {"section": str(i)},
                "options": {
                    "min_counts": 100,
                    "always_filter": [],
                    "always_keep": [1],
                },
            }
            for i, slide in enumerate(slides)
        }
    )

    with open(target, "w") as fp:
        fp.write(config.as_string())


cli.add_command(init)


@click.command()
@click.argument(
    "project-file",
    type=click.Path(dir_okay=False, readable=True, resolve_path=True),
)
@click.option("--session", type=click.File("rb"))
@click.option("--tensorboard/--no-tensorboard", default=True)
@click.option("--stats/--no-stats", default=False)
@click.option("--stats-conditions-interval", default=10)
@click.option("--stats-elbo-interval", default=10)
@click.option("--stats-image-interval", default=1000)
@click.option("--stats-latent-interval", default=1000)
@click.option("--stats-metagenefullsummary-interval", default=0)
@click.option("--stats-metagenehistogram-interval", default=0)
@click.option("--stats-metagenemean-interval", default=100)
@click.option("--stats-metagenesummary-interval", default=1000)
@click.option("--stats-rmse-interval", default=10)
@click.option("--stats-scale-interval", default=1000)
@click.option("--checkpoint-interval", default=1000)
@click.option("--purge-interval", default=1000)
@click.option("--analysis-interval", default=0)
@_init
def run(
    project_file,
    session,
    tensorboard,
    stats,
    stats_conditions_interval,
    stats_elbo_interval,
    stats_image_interval,
    stats_latent_interval,
    stats_metagenefullsummary_interval,
    stats_metagenehistogram_interval,
    stats_metagenemean_interval,
    stats_metagenesummary_interval,
    stats_rmse_interval,
    stats_scale_interval,
    checkpoint_interval,
    purge_interval,
    analysis_interval,
):
    r"""
    Runs xfuse based on a project configuration file.
    The configuration file can be created manually or using the `init`
    subcommand.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    base_session = load_session(session) if session is not None else Session()
    with base_session:
        with open(project_file) as fp:
            config = dict(tomlkit.loads(fp.read()))
        config = merge_config(config)

        def _expand_path(path):
            path = os.path.expanduser(path)
            if os.path.isabs(path):
                return path
            return os.path.join(os.path.dirname(project_file), path)

        for name, slide in config["slides"].items():
            try:
                data_path = slide["data"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Slide {name} does not have a `data` attribute"
                ) from exc
            config["slides"][name]["data"] = _expand_path(data_path)

        with open(first_unique_filename("merged_config.toml"), "w") as f:
            f.write(tomlkit.dumps(config))

        slide_paths = {
            name: slide["data"] for name, slide in config["slides"].items()
        }
        slide_options = {
            name: slide["options"] if "options" in slide else {}
            for name, slide in config["slides"].items()
        }
        slide_covariates = {
            name: {
                covariate: str(condition)
                for covariate, condition in slide["covariates"].items()
            }
            if "covariates" in slide
            else {}
            for name, slide in config["slides"].items()
        }

        covariates = {
            **{
                covariate: sorted(set(x[1] for x in group))
                for covariate, group in it.groupby(
                    sorted(
                        (covariate, condition)
                        for slide, covariates in slide_covariates.items()
                        for covariate, condition in covariates.items()
                    ),
                    key=lambda x: x[0],
                )
            },
            **get("covariates"),
        }

        design = pd.DataFrame(
            index=slide_covariates.keys(), columns=covariates.keys()
        )
        for slide_name, slide_covariates in slide_covariates.items():
            for slide_covariate, slide_condition in slide_covariates.items():
                design.loc[slide_name, slide_covariate] = slide_condition
        design = design.astype("category")
        for covariate, conditions in covariates.items():
            design[covariate].cat.set_categories(conditions)

        log(INFO, "Using the following design table:")
        log(INFO, "")
        for x in tabulate(
            design.astype(object).fillna("<inferred>"),
            headers=[
                "{} {{{}}}".format(c, ",".join(covariates[c]))
                for c in design.columns
            ],
            tablefmt="orgtbl",
        ).split("\n"):
            log(INFO, x)
        log(INFO, "")

        expansion_strategy = get("metagene_expansion_strategy")
        if expansion_strategy is None:
            expansion_strategy = expansion_strategies[
                config["expansion_strategy"]["type"]
            ](
                **config["expansion_strategy"][
                    config["expansion_strategy"]["type"]
                ]
            )

        stats_writers = []
        if stats:
            stats_writers.append(FileWriter())
        if tensorboard:
            stats_writers.append(TensorboardWriter())

        analyses = {
            name: (settings["type"], settings["options"])
            for name, settings in config["analyses"].items()
        }

        def _every(n):
            def _predicate(**_msg):
                return not get("eval") and get("training_data").step % n == 0

            return _predicate

        messengers = []
        if stats_conditions_interval > 0:
            messengers.append(
                stats_trackers.Conditions(_every(stats_conditions_interval))
            )
        if stats_elbo_interval > 0:
            messengers.append(stats_trackers.ELBO(_every(stats_elbo_interval)))
        if stats_image_interval > 0:
            messengers.append(
                stats_trackers.Image(_every(stats_image_interval))
            )
        if stats_latent_interval > 0:
            messengers.append(
                stats_trackers.Latent(_every(stats_latent_interval))
            )
        if stats_metagenefullsummary_interval > 0:
            messengers.append(
                stats_trackers.MetageneFullSummary(
                    _every(stats_metagenefullsummary_interval)
                )
            )
        if stats_metagenehistogram_interval > 0:
            messengers.append(
                stats_trackers.MetageneHistogram(
                    _every(stats_metagenehistogram_interval)
                )
            )
        if stats_metagenemean_interval > 0:
            messengers.append(
                stats_trackers.MetageneMean(
                    _every(stats_metagenemean_interval)
                )
            )
        if stats_metagenesummary_interval > 0:
            messengers.append(
                stats_trackers.MetageneSummary(
                    _every(stats_metagenesummary_interval)
                )
            )
        if stats_rmse_interval > 0:
            messengers.append(stats_trackers.RMSE(_every(stats_rmse_interval)))
        if stats_scale_interval > 0:
            messengers.append(
                stats_trackers.Scale(_every(stats_scale_interval))
            )
        if checkpoint_interval > 0:
            messengers.append(Checkpointer(period=checkpoint_interval))
        if analysis_interval > 0:
            messengers.append(
                AnalysisRunner(analyses=analyses, period=analysis_interval)
            )

        with Session(
            covariates=covariates,
            messengers=messengers,
            stats_writers=stats_writers,
        ):
            _run(
                design,
                slide_paths,
                analyses=analyses,
                expansion_strategy=expansion_strategy,
                purge_interval=purge_interval,
                network_depth=config["xfuse"]["network_depth"],
                network_width=config["xfuse"]["network_width"],
                min_counts=config["xfuse"]["min_counts"],
                gene_regex=config["xfuse"]["gene_regex"],
                patch_size=config["optimization"]["patch_size"],
                batch_size=config["optimization"]["batch_size"],
                epochs=config["optimization"]["epochs"],
                learning_rate=config["optimization"]["learning_rate"],
                cache_data=config["settings"]["cache_data"],
                num_data_workers=config["settings"]["data_workers"],
                slide_options=slide_options,
            )


cli.add_command(run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
