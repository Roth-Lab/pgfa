import numpy as np

from pgfa.math_utils import discrete_rvs_gumbel_trick

from pgfa.updates.base import FeatureAllocationMatrixUpdater


class GibbsUpdater(FeatureAllocationMatrixUpdater):

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        log_p = np.zeros(2)

        for k in cols:
            params.Z[row_idx, k] = 0

            log_p[0] = np.log1p(-feat_probs[k])
            log_p[0] += dist.log_p_row(data, params, row_idx)

            params.Z[row_idx, k] = 1

            log_p[1] = np.log(feat_probs[k])
            log_p[1] += dist.log_p_row(data, params, row_idx)

            params.Z[row_idx, k] = discrete_rvs_gumbel_trick(log_p)

        return params
