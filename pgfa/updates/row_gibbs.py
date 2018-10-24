import itertools
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize

from pgfa.updates.base import FeatureAllocationMatrixUpdater


class RowGibbsUpdater(FeatureAllocationMatrixUpdater):
    def __init__(self, max_cols=None, singletons_updater=None):
        self.max_cols = max_cols

        self.singletons_updater = singletons_updater

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        if len(cols) == 0:
            return params

        max_cols = len(cols)

        if self.max_cols is not None:
            max_cols = min(self.max_cols, max_cols)

        update_cols = np.random.choice(cols, replace=False, size=max_cols)

        Zs = np.tile(params.Z[row_idx], (2**max_cols, 1))

        Zs[:, update_cols] = list(map(np.array, itertools.product([0, 1], repeat=max_cols)))

        Zs = np.array(Zs, dtype=np.int)

        return do_row_gibbs_update(
            cols,
            data,
            dist,
            feat_probs,
            params,
            row_idx,
            Zs
        )


def do_row_gibbs_update(cols, data, dist, feat_probs, params, row_idx, Zs):
    log_p1 = np.log(feat_probs[cols])

    log_p0 = np.log(1 - feat_probs[cols])

    log_p = np.zeros(len(Zs))

    for idx in range(len(Zs)):
        params.Z[row_idx] = Zs[idx]

        log_p[idx] = np.sum(Zs[idx, cols] * log_p1) + np.sum((1 - Zs[idx, cols]) * log_p0) + \
            dist.log_p_row(data, params, row_idx)

    log_p = log_normalize(log_p)

    idx = discrete_rvs(np.exp(log_p))

    params.Z[row_idx] = Zs[idx]

    return params
