# type: ignore
# pylint: disable=isinstance-second-argument-not-valid-type
# ^ due to https://github.com/PyCQA/pylint/issues/3507

import warnings
from copy import deepcopy
from inspect import signature
from typing import Dict, List, NamedTuple, Optional, OrderedDict, Union

import tomlkit

from .analyze import analyses
from .model.experiment.st.metagene_expansion_strategy import STRATEGIES


class Item(NamedTuple):
    r"""Configuration item"""
    comment: Optional[str] = None
    value: Optional[Union["AnnotatedConfig", "Value"]] = None
    example: bool = False


Value = Union[float, List[float], str, List[str]]
Config = Dict[str, Union["Config", Value]]
AnnotatedConfig = OrderedDict[str, Item]  # pylint: disable=invalid-name


_ANNOTATED_CONFIG = OrderedDict(
    # pylint: disable=line-too-long
    [
        (
            "xfuse",
            Item(
                comment=" ".join(
                    [
                        "This section defines modeling options.",
                        "It can usually be left as-is.",
                    ]
                ),
                value=OrderedDict(
                    [
                        ("network_depth", Item(value=6)),
                        ("network_width", Item(value=16)),
                        (
                            "gene_regex",
                            Item(
                                value="^(?!RPS|RPL|MT-).*",
                                comment=" ".join(
                                    [
                                        "Regex matching genes to include in the model.",
                                        "By default, exclude mitochondrial and ribosomal genes.",
                                    ]
                                ),
                            ),
                        ),
                        (
                            "min_counts",
                            Item(
                                value=1,
                                comment="Exclude all genes with fewer reads than this value.",
                            ),
                        ),
                    ]
                ),
            ),
        ),
        (
            "settings",
            Item(
                value=OrderedDict(
                    [
                        (
                            "cache_data",
                            Item(
                                value=True,
                                comment=" ".join(
                                    [
                                        "If true, always keep data in memory (better performance).",
                                        "Otherwise, load data selectively for each mini-batch (lower memory usage).",
                                    ]
                                ),
                            ),
                        ),
                        (
                            "data_workers",
                            Item(
                                value=8,
                                comment=" ".join(
                                    [
                                        "Number of worker processes for data loading.",
                                        "If set to zero, run data loading in main thread.",
                                    ]
                                ),
                            ),
                        ),
                    ]
                )
            ),
        ),
        (
            "expansion_strategy",
            Item(
                comment="This section contains configuration options for the metagene expansion strategy.",
                value=OrderedDict(
                    [
                        (
                            "type",
                            Item(
                                comment=f"Available choices: {', '.join(STRATEGIES.keys())}",
                                value="DropAndSplit",
                            ),
                        ),
                        (
                            "purge_interval",
                            Item(
                                comment="Metagene purging interval (epochs)",
                                value=1000,
                            ),
                        ),
                        *[
                            (
                                name,
                                Item(
                                    value=OrderedDict(
                                        [
                                            (
                                                param_name,
                                                Item(value=param.default),
                                            )
                                            for param_name, param in signature(
                                                strategy
                                            ).parameters.items()
                                        ]
                                    )
                                ),
                            )
                            for name, strategy in STRATEGIES.items()
                        ],
                    ]
                ),
            ),
        ),
        (
            "optimization",
            Item(
                comment=" ".join(
                    [
                        "This section defines options used during training.",
                        "It may be necessary to decrease the batch or patch size if running out of memory during training.",
                    ]
                ),
                value=OrderedDict(
                    [
                        ("batch_size", Item(value=4)),
                        ("epochs", Item(value=20000)),
                        ("learning_rate", Item(value=3e-4)),
                        (
                            "patch_size",
                            Item(
                                comment=" ".join(
                                    [
                                        "Size of training patches.",
                                        "Set to '-1' to use as large patches as possible.",
                                    ]
                                ),
                                value=768,
                            ),
                        ),
                    ]
                ),
            ),
        ),
        (
            "analyses",
            Item(
                comment=" ".join(
                    [
                        "This section defines which analyses to run.",
                        "Each analysis has its own subtable with configuration options.",
                        "Remove the table to stop the analysis from being run.",
                    ]
                ),
                example=True,
                value=OrderedDict(
                    [
                        (
                            f"analysis-{name}",
                            Item(
                                comment=analysis.description,
                                value=OrderedDict(
                                    [
                                        ("type", Item(value=name)),
                                        (
                                            "options",
                                            Item(
                                                value=OrderedDict(
                                                    [
                                                        (
                                                            param_name,
                                                            Item(
                                                                value=param.default
                                                            ),
                                                        )
                                                        for param_name, param in signature(
                                                            analysis.function
                                                        ).parameters.items()
                                                    ]
                                                )
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        )
                        for name, analysis in analyses.items()
                    ]
                ),
            ),
        ),
        (
            "slides",
            Item(
                comment=" ".join(
                    [
                        "This section defines the slides to use in the experiment.",
                        'Covariates are specified in the "covariates" table.',
                        'Slide-specific options can be specified in the "options" table.',
                    ]
                ),
                value=OrderedDict([]),
                example=True,
            ),
        ),
    ]
)


def _annotated_config2config(x: AnnotatedConfig) -> Config:
    if not isinstance(x, Dict):
        return deepcopy(x)
    return {k: _annotated_config2config(v.value) for k, v in x.items()}


def construct_default_config() -> Config:
    r"""
    Provides the default configuration as a :class:`Config`
    """
    return _annotated_config2config(_ANNOTATED_CONFIG)


def _annotated_config2toml(
    x: AnnotatedConfig,
) -> tomlkit.toml_document.TOMLDocument:
    def _add_items(
        table: Union[tomlkit.toml_document.TOMLDocument, tomlkit.items.Table],
        items: OrderedDict[str, Item],
    ):
        for k, item in items.items():
            if isinstance(item.value, OrderedDict):
                subtable = tomlkit.table()
                if item.comment:
                    subtable.add(tomlkit.comment(item.comment))
                _add_items(subtable, item.value)
                table.add(k, subtable)
            else:
                try:
                    table.add(k, item.value)
                except ValueError:
                    continue
                if item.comment:
                    try:
                        table[k].comment(item.comment)
                    except AttributeError:
                        # TODO
                        pass

    config = tomlkit.document()
    _add_items(config, x)
    return config


def construct_default_config_toml() -> tomlkit.toml_document.TOMLDocument:
    r"""
    Provides the default configuration as a
    :class:`~tomlkit.toml_document.TOMLDocument`
    """
    return _annotated_config2toml(_ANNOTATED_CONFIG)


def merge_config(config: Config) -> Config:
    r"""
    Merges `config` with the default config by filling in missing keys from the
    latter.

    :raises `RuntimeError`: If `config` contains a :class:`Value` where there
    should be a :class:`Config`.
    """

    def _merge(a: Config, b: AnnotatedConfig) -> None:
        for k in a:
            if k not in b:
                warnings.warn(f'Unrecognized configuration option "{k}"')
            else:
                if isinstance(b[k].value, OrderedDict) and not b[k].example:
                    if not isinstance(a[k], Dict):
                        raise RuntimeError(
                            f'Configuration option "{k}" is misspecified'
                        )
                    _merge(a[k], b[k].value)
        for k in b:
            if k not in a:
                if b[k].example:
                    a[k] = {}
                else:
                    a[k] = _annotated_config2config(b[k].value)

    config = config.copy()
    _merge(config, _ANNOTATED_CONFIG)
    return config
