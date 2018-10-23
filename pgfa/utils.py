import numpy as np
import time


def feature_mat_to_dict(Z):
    Z_dict = {}

    for n in range(Z.shape[0]):
        Z_dict[n] = set(np.where(Z[n] == 1)[0])

    return Z_dict


def get_b_cubed_score(features_true, features_pred):
    features_true = feature_mat_to_dict(features_true)

    features_pred = feature_mat_to_dict(features_pred)

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


class Timer:
    """ Taken from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html
    """

    def __init__(self, func=time.time):
        self.elapsed = 0.0

        self._func = func

        self._start = None

    @property
    def running(self):
        return self._start is not None

    def reset(self):
        self.elapsed = 0.0

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')

        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')

        end = self._func()

        self.elapsed += end - self._start

        self._start = None

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()
