import itertools as it

import os


def unique_prefix(prefix):
    for path in it.chain(
            (prefix,),
            (f'{prefix}.{i}' for i in it.count(1)),
    ):
        if not os.path.exists(path):
            return path
