import numpy as np


def get_b_cubed_score(features_true, features_pred):
    n = len(features_pred)

    p = []

    r = []

    for i in range(n):
        for j in range(i, n):

            c = len(set(features_pred[i]) & set(features_pred[j]))

            l = len(set(features_true[i]) & set(features_true[j]))

            num = min(c, l)

            if c > 0:
                p.append(num / c)

            if l > 0:
                r.append(num / l)

    p = sum(p) / len(p)

    r = sum(r) / len(r)

    f = 2 * (p * r) / (p + r)

    return f, p, r


def lof_argsort(Z):
    return np.argsort(
        np.apply_along_axis(
            to_binary, 0, Z
        )
    )[::-1]


def lof_sort(Z):
    idxs = lof_argsort(Z)

    return Z[:, idxs]


def to_binary(x):
    value = 0

    for i, x_i in enumerate(reversed(x)):
        value += x_i * 2 ** i

    return value
