# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

import itertools as it
import json
import logging
import os
import sys
import warnings
from contextlib import ExitStack
from datetime import datetime as dt
from functools import wraps

import click
import h5py
import numpy as np
import pandas as pd
import tomlkit

from imageio import imread
from PIL import Image

from . import __version__, convert
from ._config import (  # type: ignore
    construct_default_config_toml,
    merge_config,
)
from .logging import DEBUG, INFO, WARNING, log
from .model.experiment.st.metagene_expansion_strategy import (
    STRATEGIES as expansion_strategies,
)
from .run import run as _run
from .session import Session, get
from .utility import design_matrix_from, temp_attr, with_
from .utility.file import first_unique_filename
from .session.io import load_session


_DEFAULT_SESSION = Session()

sys.excepthook = lambda *_: None


def _init(f):
    @wraps(f)
    @with_(_DEFAULT_SESSION)
    def _wrapped(*args, **kwargs):
        logging.captureWarnings(True)
        return f(*args, **kwargs)

    return _wrapped


@click.group()
@click.option("--debug", is_flag=True)
@click.version_option()
def cli(debug):
    if debug:
        _DEFAULT_SESSION.log_level = -999


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
@click.option("--rotate/--no-rotate", default=False)
@click.option(
    "--output-file",
    type=click.Path(exists=False, writable=True),
    required=True,
)
@_init
def _convert_visium(
    image,
    bc_matrix,
    tissue_positions,
    annotation,
    scale_factors,
    scale,
    mask,
    rotate,
    output_file,
):
    r"""Converts 10X Visium data"""
    tissue_positions = pd.read_csv(tissue_positions, index_col=0, header=None)
    tissue_positions = tissue_positions[[4, 5]]
    tissue_positions = tissue_positions.rename(columns={4: "y", 5: "x"})

    scale_factors = json.load(scale_factors)
    spot_radius = scale_factors["spot_diameter_fullres"] / 2

    with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
        image_data = imread(image)

    if annotation:
        with h5py.File(annotation, "r") as annotation_file:
            annotation = {
                k: annotation_file[k][()] for k in annotation_file.keys()
            }

    with h5py.File(bc_matrix, "r") as data:
        convert.visium.run(
            image_data,
            data,
            tissue_positions,
            spot_radius,
            output_file,
            annotation=annotation,
            scale_factor=scale,
            mask=mask,
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
@click.option("--rotate/--no-rotate", default=False)
@click.option(
    "--output-file",
    type=click.Path(exists=False, writable=True),
    required=True,
)
@_init
def _convert_st(
    counts,
    image,
    spots,
    transformation_matrix,
    annotation,
    scale,
    mask,
    rotate,
    output_file,
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
    convert.st.run(
        counts_data,
        image_data,
        output_file,
        spots=spots_data,
        transformation=transformation,
        annotation=annotation,
        scale_factor=scale,
        mask=mask,
        rotate=rotate,
    )


_convert.add_command(_convert_st, "st")


@click.command()
@click.option("--image", type=click.File("rb"), required=True)
@click.option("--annotation", type=click.File("rb"))
@click.option("--scale", type=float)
@click.option("--mask/--no-mask", default=True)
@click.option("--rotate/--no-rotate", default=False)
@click.option(
    "--output-file",
    type=click.Path(exists=False, writable=True),
    required=True,
)
@_init
def _convert_image(
    image, annotation, scale, mask, rotate, output_file,
):
    r"""Converts image without any associated expression data"""
    with temp_attr(Image, "MAX_IMAGE_PIXELS", None):
        image_data = imread(image)
    if annotation:
        with h5py.File(annotation, "r") as annotation_file:
            annotation = {
                k: annotation_file[k][()] for k in annotation_file.keys()
            }
    convert.image.run(
        image_data,
        output_file,
        annotation=annotation,
        scale_factor=scale,
        mask=mask,
        rotate=rotate,
    )


_convert.add_command(_convert_image, "image")


@click.command()
@click.argument("target", type=click.Path(), default=f"{__package__}.toml")
@click.argument(
    "slides", type=click.Path(exists=True, dir_okay=False), nargs=-1
)
@_init
def init(target, slides):
    r"""Creates a template for the project configuration file."""
    config = construct_default_config_toml()
    if len(slides) > 0:
        config["slides"] = {slide: {} for slide in slides}
    with open(target, "w") as fp:
        fp.write(config.as_string())


cli.add_command(init)


@click.command()
@click.argument("project-file", type=click.File("rb"))
@click.option(
    "--save-path",
    type=click.Path(),
    default=f"xfuse-{dt.now().isoformat()}",
    help="The output path",
    show_default=True,
)
@click.option("--session", type=click.File("rb"))
@_init
def run(project_file, save_path, session):
    r"""
    Runs xfuse based on a project configuration file.
    The configuration file can be created manually or using the `init`
    subcommand.
    """
    base_session = load_session(session) if session is not None else Session()
    os.makedirs(save_path, exist_ok=True)
    with open(
        first_unique_filename(os.path.join(save_path, "log")), "w"
    ) as log_file, base_session, Session(
        save_path=save_path, log_file=[sys.stderr, log_file]
    ):
        config = dict(tomlkit.loads(project_file.read().decode()))
        config = merge_config(config)

        def _expand_path(path):
            path = os.path.expanduser(path)
            if os.path.isabs(path):
                return path
            return os.path.join(os.path.dirname(project_file.name), path)

        config["slides"] = {
            _expand_path(filename): v
            for filename, v in config["slides"].items()
        }

        if config["xfuse"]["version"] != __version__:
            log(
                WARNING,
                "Config was created using %s version %s"
                " but this is version %s",
                __package__,
                config["xfuse"]["version"],
                __version__,
            )
            config["xfuse"]["version"] = __version__

        with open(
            first_unique_filename(
                os.path.join(save_path, "merged_config.toml")
            ),
            "w",
        ) as f:
            f.write(tomlkit.dumps(config))

        slide_options = {
            filename: slide["options"] if "options" in slide else {}
            for filename, slide in config["slides"].items()
        }
        design = design_matrix_from(
            {
                filename: {k: v for k, v in slide.items() if k != "options"}
                for filename, slide in config["slides"].items()
            },
            covariates=get("covariates"),
        )
        covariates = [
            (k, [x for _, x in v])
            for k, v in it.groupby(design.index, key=lambda x: x[0])
        ]
        log(INFO, "Using the following design covariates:")
        for name, values in covariates:
            log(INFO, "  - %s: %s", name, ", ".join(map(str, values)))

        expansion_strategy = get("metagene_expansion_strategy")
        if expansion_strategy is None:
            expansion_strategy = expansion_strategies[
                config["expansion_strategy"]["type"]
            ](
                **config["expansion_strategy"][
                    config["expansion_strategy"]["type"]
                ]
            )

        genes = get("genes")
        if genes is None and config["xfuse"]["genes"] != []:
            genes = config["xfuse"]["genes"]

        with Session(covariates=covariates, genes=genes):
            _run(
                design,
                analyses=config["analyses"],
                expansion_strategy=expansion_strategy,
                network_depth=config["xfuse"]["network_depth"],
                network_width=config["xfuse"]["network_width"],
                min_counts=config["xfuse"]["min_counts"],
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
