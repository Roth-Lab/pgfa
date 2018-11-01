import numba
import numpy as np
import time


@numba.njit
def get_b_cubed_score(features_true, features_pred):
    n = len(features_pred)

    size = (n * (n + 1)) // 2

    p = np.zeros(size)

    r = np.zeros(size)

    idx = 0

    for i in range(n):
        for j in range(i, n):
            c = np.sum(np.logical_and(features_pred[i] == 1, features_pred[j] == 1))

            l = np.sum(np.logical_and(features_true[i] == 1, features_true[j] == 1))

            num = min(c, l)

            if c > 0:
                p[idx] = num / c

            else:
                p[idx] = np.nan

            if l > 0:
                r[idx] = num / l

            else:
                r[idx] = np.nan

            idx += 1

    p = np.nanmean(p)

    r = np.nanmean(r)

    f = 2 * (p * r) / max((p + r), 1)

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
