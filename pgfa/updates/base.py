import numpy as np


class FeatureAllocationMatrixUpdater(object):
    def __init__(self, singletons_updater=None):
        self.singletons_updater = singletons_updater

    def update(self, model):
        num_rows = model.params.Z.shape[0]

        for row_idx in np.random.permutation(num_rows):
            cols = model.feat_alloc_dist.get_update_cols(model.params, row_idx)

            feat_probs = model.feat_alloc_dist.get_feature_probs(model.params, row_idx)

            model.params = self.update_row(cols, model.data, model.data_dist, feat_probs, model.params, row_idx)

            if self.singletons_updater is not None:
                self.singletons_updater.update_row(model, row_idx)

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        raise NotImplementedError
