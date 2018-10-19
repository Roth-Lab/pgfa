import numpy as np

from pgfa.math_utils import discrete_rvs, log_normalize

from pgfa.updates.base import FeatureAllocationMatrixUpdater


class GibbsUpdater(FeatureAllocationMatrixUpdater):
    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        log_p = np.zeros(2)

        for k in cols:
            params.Z[row_idx, k] = 0

            log_p[0] = np.log(1 - feat_probs[k])
            log_p[0] += dist.log_p_row(data, params, row_idx)

            params.Z[row_idx, k] = 1

            log_p[1] = np.log(feat_probs[k])
            log_p[1] += dist.log_p_row(data, params, row_idx)

            log_p = log_normalize(log_p)

            params.Z[row_idx, k] = discrete_rvs(np.exp(log_p))

        return params
