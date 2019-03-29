import numba
import numpy as np
import time

import pgfa.updates


def get_feat_alloc_updater(mixed_updates=False, updater='g', updater_kwargs={}):
    if updater == 'dpf':   
        feat_alloc_updater = pgfa.updates.DicreteParticleFilterUpdater(**updater_kwargs)
    
    elif updater == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(**updater_kwargs)
        
    elif updater == 'gpf':
        feat_alloc_updater = pgfa.updates.GumbelParticleFilterUpdater(**updater_kwargs)

    elif updater == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(**updater_kwargs)

    elif updater == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(**updater_kwargs)
    
    else:
        raise Exception('Unrecognized feature allocation updater: {}'.format(updater))
    
    if mixed_updates:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

    return feat_alloc_updater


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)

        set_numba_seed(seed)

    
@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


@numba.njit
def summarize_feature_allocation_matrix(Zs, burnin=0, thin=1):
    I = len(Zs)
    
    Zs = Zs[:burnin:thin]
    
    best_score = 0
    
    best_Z = Zs[0]
    
    for i in range(I):
        score = 0
        
        for j in range(I):
            score += get_b_cubed_score(Zs[i], Zs[j])[0]
        
        score /= I
        
        if score > best_score:
            best_score = score
            
            best_Z = Zs[i]
    
    return best_Z


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
