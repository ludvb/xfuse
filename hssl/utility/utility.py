import random
import re
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    cast,
    overload,
)

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as ss
import torch

from ..logging import INFO, log
from ..session import get
from ..utility.file import extension

__all__ = [
    "compose",
    "set_rng_seed",
    "center_crop",
    "read_data",
    "design_matrix_from",
    "find_device",
    "sparseonehot",
    "to_device",
    "with_",
]


def compose(f: Callable[..., Any], *gs: Callable[..., Any]):
    r"""Composes/threads given functions"""
    return (
        # pylint: disable=no-value-for-parameter
        (lambda *args, **kwargs: f(compose(*gs)(*args, **kwargs)))
        if gs != ()
        else f
    )


def set_rng_seed(seed: int) -> None:
    r"""
    Sets the seed of the :module:`random`, :module:`numpy`, and :module:`torch`
    RNGs
    """
    i32max = np.iinfo(np.int32).max
    random.seed(seed)
    n_seed = random.choice(range(i32max + 1))
    t_seed = random.choice(range(i32max + 1))
    np.random.seed(n_seed)
    torch.manual_seed(t_seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    log(
        INFO,
        " / ".join(
            [
                "random rng seeded with %d",
                "numpy rng seeded with %d",
                "torch rng seeded with %d",
            ]
        ),
        seed,
        n_seed,
        t_seed,
    )


def center_crop(x, target_shape):
    r"""Crops `x` to the given `target_shape` from the center"""
    return x[
        tuple(
            [
                slice((a - b) // 2, (a - b) // 2 + b)
                if b is not None
                else slice(None)
                for a, b in zip(x.shape, target_shape)
            ]
        )
    ]


def read_data(
    paths: List[str],
    filter_ambiguous: bool = True,
    num_genes: Optional[int] = None,
    genes: List[str] = None,
) -> pd.DataFrame:
    r"""Reads data from `paths` and produces the (concatenated) data table"""

    def _load_file(path: str) -> pd.DataFrame:
        def _csv(sep):
            return pd.read_csv(path, sep=sep, index_col=0).astype(
                pd.SparseDtype(np.float32, 0)
            )

        def _h5():
            with h5py.File(path, "r") as data:
                spmatrix = ss.csr_matrix(
                    (
                        data["matrix"]["data"],
                        data["matrix"]["indices"],
                        data["matrix"]["indptr"],
                    )
                )
                return pd.DataFrame.sparse.from_spmatrix(
                    spmatrix.astype(np.float32),
                    columns=data["matrix"]["columns"],
                    index=pd.Index(data["matrix"]["index"], name="n"),
                )

        parse_dict: Dict[str, Callable[[], pd.DataFrame]] = {
            r"csv(:?\.(gz|bz2))$": partial(_csv, sep=","),
            r"tsv(:?\.(gz|bz2))$": partial(_csv, sep="\t"),
            r"hdf(:?5)$": _h5,
            r"he5$": _h5,
            r"h5$": _h5,
        }

        log(INFO, "loading data file %s", path)
        for pattern, parser in parse_dict.items():
            if re.match(pattern, extension(path)):  # type: ignore
                return parser()
        raise NotImplementedError(
            f"no parser for file {path}"
            f" (supported file extension patterns:"
            f' {", ".join(parse_dict.keys())})'
        )

    ks, xs = zip(*[(p, _load_file(p)) for p in paths])
    data = pd.concat(xs, keys=ks, join="outer", axis=0, sort=False).fillna(0.0)

    data = data.iloc[:, (data.sum(0) > 0).values]

    if genes is not None:
        data_ = pd.DataFrame(
            np.zeros((len(data), len(genes))),
            columns=genes,
            index=data.index,
            dtype=float,
        )
        shared_genes = np.intersect1d(genes, data.columns)
        data_[shared_genes] = data[shared_genes]
        data = data_
    elif filter_ambiguous:
        data = data[[x for x in data.columns if "ambiguous" not in x]]

    if num_genes:
        if isinstance(num_genes, int):
            # pylint: disable=invalid-unary-operand-type
            data = data[data.sum(0).sort_values()[-num_genes:].index]
        if isinstance(num_genes, list):
            data = data[num_genes]

    return data


def design_matrix_from(
    design: pd.DataFrame,
    covariates: Optional[List[Tuple[str, Set[str]]]] = None,
) -> pd.DataFrame:
    r"""
    Constructs the design matrix from the design specified in the design file
    """

    if len(design.columns) == 0:
        return pd.DataFrame(np.zeros((0, len(design))))

    design = (
        design[list(sorted(design.columns))].astype(str).astype("category")
    )

    if covariates is not None:
        missing_covariates = [
            x for x, _ in covariates if x not in design.columns
        ]
        if missing_covariates != []:
            raise ValueError(
                "the following covariates are missing from the design: "
                + ", ".join(missing_covariates)
            )

        for covariate, values in covariates:
            design[covariate].cat.set_categories(sorted(values), inplace=True)
        design = design[[x for x, _ in covariates]]
    else:
        for covariate in design.columns:
            design[covariate].cat.set_categories(
                sorted(design[covariate].cat.categories), inplace=True
            )

    def _encode(covariate):
        log(
            INFO,
            'encoding design covariate "%s" with %d categories: %s',
            covariate.name,
            len(covariate.cat.categories),
            ", ".join(covariate.cat.categories),
        )
        return pd.DataFrame(
            (
                np.eye(len(covariate.cat.categories), dtype=int)[
                    :, covariate.cat.codes
                ]
            ),
            index=covariate.cat.categories,
        )

    ks, vs = zip(*[(k, _encode(v)) for k, v in design.iteritems()])
    return pd.concat(vs, keys=ks)


def find_device(x: Any) -> torch.device:
    r"""
    Tries to find the :class:`torch.device` associated with the given object
    """

    class NoDevice(Exception):
        # pylint: disable=missing-class-docstring
        pass

    if isinstance(x, torch.Tensor):
        return x.device

    if isinstance(x, list):
        for y in x:
            try:
                return find_device(y)
            except NoDevice:
                pass

    if isinstance(x, dict):
        for y in x.values():
            try:
                return find_device(y)
            except NoDevice:
                pass

    raise NoDevice(f"Failed to find a device associated with {x}")


def sparseonehot(labels: torch.Tensor, num_classes: Optional[int] = None):
    r"""One-hot encodes a label vectors into a sparse tensor"""
    if num_classes is None:
        num_classes = cast(int, labels.max().item()) + 1
    idx = torch.stack([torch.arange(labels.shape[0]).to(labels), labels])
    return torch.sparse.LongTensor(  # type: ignore
        idx,
        torch.ones(idx.shape[1]).to(idx),
        torch.Size([labels.shape[0], num_classes]),
    )


@overload
def to_device(
    x: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: List[Any], device: Optional[torch.device] = None
) -> List[Any]:
    # pylint: disable=missing-function-docstring
    ...


@overload
def to_device(
    x: Dict[Any, Any], device: Optional[torch.device] = None
) -> Dict[Any, Any]:
    # pylint: disable=missing-function-docstring
    ...


def to_device(x, device=None):
    r"""
    Converts :class:`torch.Tensor` or a collection of :class:`torch.Tensor` to
    the given :class:`torch.device`
    """
    if device is None:
        device = get("default_device")
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(y, device) for y in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    raise NotImplementedError()


def with_(ctx: ContextManager) -> Callable[[Callable], Callable]:
    r"""
    Creates a decorator that runs the decorated function in the given context
    manager
    """

    def _decorator(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            with ctx:
                return f(*args, **kwargs)

        return _wrapped

    return _decorator
