import numpy as np


class FeatureAllocationMatrixUpdater(object):

    def __init__(self, annealing_schedule=None, singletons_updater=None):
        self.annealing_schedule = annealing_schedule

        self.singletons_updater = singletons_updater

        self.iter = 0

    def update(self, model):
        self.iter += 1

        if self.annealing_schedule is not None:
            model.data_dist.annealing_power = self.annealing_schedule(self.iter)

        num_rows = model.params.Z.shape[0]

        for row_idx in np.random.permutation(num_rows):
            cols = model.feat_alloc_dist.get_update_cols(model.params, row_idx)

            if len(cols) > 0:
                feat_probs = model.feat_alloc_dist.get_feature_probs(model.params, row_idx)

                model.params = self.update_row(cols, model.data, model.data_dist, feat_probs, model.params, row_idx)

            if self.singletons_updater is not None:
                self.singletons_updater.update_row(model, row_idx)

        model.data_dist.annealing_power = 1.0

    def update_row(self, cols, data, dist, feat_probs, params, row_idx):
        raise NotImplementedError
