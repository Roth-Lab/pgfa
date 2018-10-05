import itertools
import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize

from .utils import get_rows


class RowGibbsUpdater(object):
    def __init__(self, max_cols=None):
        self.max_cols = max_cols

    def update(self, data, dist, feat_alloc_prior, params):
        for row_idx in get_rows(params.N):
            cols = feat_alloc_prior.get_update_cols(row_idx, params.Z)

            feat_probs = feat_alloc_prior.get_feature_probs(row_idx, params.Z)

            if self.max_cols is None:
                max_cols = len(cols)

            else:
                max_cols = self.max_cols

            params = do_row_gibbs_update(
                data,
                dist,
                cols,
                feat_probs,
                params,
                row_idx,
                max_cols=max_cols
            )

        return params


def do_row_gibbs_update(data, dist, cols, feat_probs, params, row_idx, max_cols=1):
    K = len(cols)

    if K > max_cols:
        K = max_cols

    update_cols = np.random.choice(cols, replace=False, size=K)

    Zs = np.tile(params.Z[row_idx], (2**K, 1))

    Zs[:, update_cols] = list(map(np.array, itertools.product([0, 1], repeat=K)))

    Zs = np.array(Zs, dtype=np.int)

    return _do_row_gibbs_update(data, dist, cols, feat_probs, params, row_idx, Zs)


@numba.jit
def _do_row_gibbs_update(data, dist, cols, feat_probs, params, row_idx, Zs):
    log_p1 = np.log(feat_probs)

    log_p0 = np.log(1 - feat_probs)

    log_p = np.zeros(len(Zs))

    z = np.ones(params.K)

    for idx in range(len(Zs)):
        # Work around because Numba won't allow params.Z[row_idx, cols] = Zs[idx]
        z[cols] = Zs[idx]

        params.Z[row_idx] = z

        log_p[idx] = np.sum(Zs[idx] * log_p1) + np.sum((1 - Zs[idx]) * log_p0) + dist.log_p_row(data, params, row_idx)

    log_p = log_normalize(log_p)

    idx = discrete_rvs(np.exp(log_p))

    z[cols] = Zs[idx]

    params.Z[row_idx] = z

    return params
