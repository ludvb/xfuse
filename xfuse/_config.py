# type: ignore

from copy import deepcopy
from inspect import signature
from typing import Dict, List, NamedTuple, Optional, OrderedDict, Union

import tomlkit

from . import __version__
from .analyze import analyses
from .logging import WARNING, log
from .model.experiment.st.factor_expansion_strategy import STRATEGIES


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
                        ("network_depth", Item(value=5)),
                        ("network_width", Item(value=16)),
                        ("encode_expression", Item(value=False)),
                        (
                            "genes",
                            Item(
                                value=[],
                                comment=" ".join(
                                    [
                                        "List of genes to include in the analysis.",
                                        "An empty list will include all genes.",
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
                        (
                            "version",
                            Item(
                                value=__version__,
                                comment=" ".join(
                                    [
                                        f"This is the version of {__package__} used to create this config file.",
                                        "It is only used for record keeping.",
                                    ]
                                ),
                            ),
                        ),
                    ]
                ),
            ),
        ),
        (
            "expansion_strategy",
            Item(
                comment="This section contains configuration options for the factor expansion strategy.",
                value=OrderedDict(
                    [
                        (
                            "type",
                            Item(
                                comment=f"Available choices: {', '.join(STRATEGIES.keys())}",
                                value="ExtraBaselines",
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
                        ("epochs", Item(value=-1)),
                        ("learning_rate", Item(value=1e-4)),
                        (
                            "patch_size",
                            Item(
                                comment=" ".join(
                                    [
                                        "Size of training patches.",
                                        "Set to '-1' to use as large patches as possible.",
                                    ]
                                ),
                                value=-1,
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
                            name,
                            Item(
                                comment=analysis.description,
                                value=OrderedDict(
                                    [
                                        (param_name, Item(value=param.default))
                                        for param_name, param in signature(
                                            analysis.function
                                        ).parameters.items()
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
                        "Section headers should be paths to the slides.",
                        "Section keys are covariates that will be modeled.",
                        'Slide-specific options can be specified in the "options" table.',
                    ]
                ),
                value=OrderedDict(
                    [
                        (
                            "/path/to/slide1.h5",
                            Item(
                                value=OrderedDict(
                                    [
                                        ("Id", Item(value=1)),
                                        ("Group", Item(value="Control")),
                                        (
                                            "options",
                                            Item(
                                                value=OrderedDict(
                                                    [
                                                        (
                                                            "min_counts",
                                                            Item(value=100),
                                                        ),
                                                        (
                                                            "always_filter",
                                                            Item(value=[]),
                                                        ),
                                                        (
                                                            "always_keep",
                                                            Item(value=[1]),
                                                        ),
                                                    ]
                                                )
                                            ),
                                        ),
                                    ]
                                )
                            ),
                        ),
                        (
                            "/path/to/slide2.h5",
                            Item(
                                value=OrderedDict(
                                    [
                                        ("Id", Item(value=2)),
                                        ("Group", Item(value="Treatment")),
                                    ]
                                )
                            ),
                        ),
                    ]
                ),
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
                table.add(k, item.value)
                if item.comment:
                    table[k].comment(item.comment)

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
                log(WARNING, f'Unrecognized configuration option "{k}"')
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
