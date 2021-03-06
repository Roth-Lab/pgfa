'''
Created on 25 Jan 2017

@author: Andrew Roth
'''
import numpy as np


class MockDataDistribution(object):
    def __init__(self):
        pass

    def log_p(self, data, params):
        return 0

    def log_p_row(self, data, params, row_idx):
        return 0


class MockFeatureAllocationPrior(object):
    def __init__(self, p=1e-4):
        self.p = p

    def get_feature_probs(self, params, row_idx):
        K = params.Z.shape[1]

        return self.p * np.ones(K)

    def get_update_cols(self, params, row_idx):
        K = params.Z.shape[1]

        cols = np.arange(K)

        np.random.shuffle(cols)

        return cols

    def log_p(self, params):
        return np.sum(params.Z * np.log(self.p)) + np.sum((1 - params.Z) * np.log(1 - self.p))


class MockParams(object):
    def __init__(self, alpha, K, N):
        self.alpha = alpha

        self.Z = np.random.randint(0, 2, size=(N, K))

    @property
    def K(self):
        return self.Z.shape[1]

    @property
    def N(self):
        return self.Z.shape[0]
