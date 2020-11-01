import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def design_matrix_from(
    design: Dict[str, Dict[str, Union[int, str]]],
    covariates: Optional[List[Tuple[str, List[str]]]] = None,
) -> pd.DataFrame:
    r"""
    Constructs the design matrix from the design specified in the design file
    """
    design_table = pd.concat(
        [
            pd.DataFrame({k: [v] for k, v in v.items()}, index=[k])
            for k, v in design.items()
        ],
        sort=True,
    )

    if covariates is None:
        covariates = [
            (k, list(map(str, xs.unique().tolist())))
            for k, xs in design_table.iteritems()
        ]

    if len(covariates) == 0:
        return pd.DataFrame(
            np.zeros((0, len(design_table))), columns=design_table.index
        )

    for covariate, values in covariates:
        if covariate not in design_table:
            design_table[covariate] = 0
        design_table[covariate] = design_table[covariate].astype("category")
        design_table[covariate].cat.set_categories(values, inplace=True)
    design_table = design_table[[x for x, _ in covariates]]

    for covariate in design_table:
        if np.any(pd.isna(design_table[covariate])):
            warnings.warn(
                f'Design covariate "{covariate}" has missing values.',
            )

    def _encode(covariate):
        oh_matrix = np.eye(len(covariate.cat.categories), dtype=int)[
            :, covariate.cat.codes
        ]
        oh_matrix[:, covariate.cat.codes == -1] = 0
        return pd.DataFrame(
            oh_matrix, index=covariate.cat.categories, columns=covariate.index
        )

    ks, vs = zip(*[(k, _encode(v)) for k, v in design_table.iteritems()])
    return pd.concat(vs, keys=ks)
