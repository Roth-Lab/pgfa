import numpy as np


def get_cols(m, include_singletons=False):
    K = len(m)

    if include_singletons:
        cols = np.arange(K)

    else:
        cols = np.atleast_1d(np.squeeze(np.where(m > 0)))

    np.random.shuffle(cols)

    return cols


def get_rows(N):
    rows = np.arange(N)

    np.random.shuffle(rows)

    return rows
