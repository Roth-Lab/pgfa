import numpy as np


def to_binary(x):
    value = 0

    for i, x_i in enumerate(reversed(x)):
        value += x_i * 2 ** i

    return value


def lof_argsort(Z):
    return np.argsort(
        np.apply_along_axis(
            to_binary, 0, Z
        )
    )[::-1]


def lof_sort(Z):
    idxs = lof_argsort(Z)

    return Z[:, idxs]
