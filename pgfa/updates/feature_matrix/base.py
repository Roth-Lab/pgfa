from .utils import get_rows


class FeatureAllocationMatrixUpdater(object):
    def update(self, data, dist, feat_alloc_prior, params):
        num_rows = params.Z.shape[0]

        for row_idx in get_rows(num_rows):
            cols = feat_alloc_prior.get_update_cols(row_idx, params.Z)

            feat_probs = feat_alloc_prior.get_feature_probs(row_idx, params.Z)

            params = self.update_row(cols, data, dist, feat_probs, params, row_idx)

        return params

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        raise NotImplementedError
