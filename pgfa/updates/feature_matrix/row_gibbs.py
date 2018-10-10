import itertools
import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize

from .base import FeatureAllocationMatrixUpdater


class RowGibbsUpdater(FeatureAllocationMatrixUpdater):
    def __init__(self, max_cols=None):
        self.max_cols = max_cols

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        max_cols = len(cols)

        if self.max_cols is not None:
            max_cols = min(self.max_cols, max_cols)

        update_cols = np.random.choice(cols, replace=False, size=max_cols)

        Zs = np.tile(params.Z[row_idx], (2**max_cols, 1))

        Zs[:, update_cols] = list(map(np.array, itertools.product([0, 1], repeat=max_cols)))

        Zs = np.array(Zs, dtype=np.int)

        params = do_row_gibbs_update(
            data,
            dist,
            cols,
            feat_probs,
            params,
            row_idx,
            max_cols=max_cols
        )


@numba.jit
def do_row_gibbs_update(data, dist, cols, feat_probs, params, row_idx, Zs):
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
