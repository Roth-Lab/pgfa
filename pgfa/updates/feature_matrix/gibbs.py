import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize

from .utils import get_rows


class GibbsUpdater(object):
    def update(self, data, dist, feat_alloc_prior, params):
        for row_idx in get_rows(params.N):
            cols = feat_alloc_prior.get_update_cols(row_idx, params.Z)

            feat_probs = feat_alloc_prior.get_feature_probs(row_idx, params.Z)

            params = do_gibbs_update(data, dist, cols, feat_probs, params, row_idx)

        return params


@numba.jit
def do_gibbs_update(data, dist, cols, feat_probs, params, row_idx):
    log_p = np.zeros(2)

    for k in cols:
        params.Z[row_idx, k] = 0

        log_p[0] = np.log(1 - feat_probs[k]) + dist.log_p_row(data, params, row_idx)

        params.Z[row_idx, k] = 1

        log_p[1] = np.log(feat_probs[k]) + dist.log_p_row(data, params, row_idx)

        log_p = log_normalize(log_p)

        params.Z[row_idx, k] = discrete_rvs(np.exp(log_p))

    return params
