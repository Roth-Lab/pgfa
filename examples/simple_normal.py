import numba
import numpy as np
import scipy.stats

from pgfa.distributions import NormalDistribution, NormalGammaProductDistribution
from pgfa.distributions.base import Distribution
from pgfa.inference.feature_matrix import GibbsFeatureAllocationMatrixKernel, RowGibbsFeatureAllocationMatrixKernel, \
    ParticleGibbsFeatureAllocationMatrixKernel
from pgfa.models import FiniteFeatureAllocationModel
from pgfa.utils import lof_sort

import pgfa.distributions.normal


def main():
    X, V, Z = get_data()

    #=========================================================================
    # Gibbs
    #=========================================================================
    print('Gibbs')

    run(X, Z, V, GibbsFeatureAllocationMatrixKernel())

    print('Particle Gibbs')

    run(X, Z, V, ParticleGibbsFeatureAllocationMatrixKernel())

    print('Row Gibbs')

    run(X, Z, V, RowGibbsFeatureAllocationMatrixKernel())


def get_data():
    K = 10

    N = 10

    X = scipy.stats.norm.rvs(10, 0.01, size=(N, 1))

    V = np.ones((K, 1)) * 100

    V[:2] = 10

    Z = np.zeros((N, K), dtype=np.int)

    Z[: N // 2, 0] = 1

    Z[N // 2:, 1] = 1

    return X, V, Z


def get_model(data, init_feat_mat, init_feat_params):
    feature_dist = NormalMeanDistribution(100)

    feature_prior_dist = NormalDistribution()

    feature_prior_params = np.array([0, 1])

    feature_weight_params = np.array([0.01, 0.99])

    return FiniteFeatureAllocationModel(
        data,
        feature_dist,
        feature_prior_dist,
        feature_prior_params,
        feature_weight_params=feature_weight_params,
        feature_params=init_feat_params,
        latent_values=init_feat_mat,
        num_features=init_feat_mat.shape[1]
    )


def run(data, init_feat_mat, init_feat_params, updater):
    model = get_model(data, init_feat_mat, init_feat_params)

    for i in range(200):
        if i % 100 == 0:
            print(i, model.log_p)

        model.latent_values = updater.update(model)

    print(lof_sort(model.latent_values))


class NormalMeanDistribution(Distribution):
    def __init__(self, precision):
        self.dist = NormalDistribution()

        self.precision = precision

        self._grad_log_p_wrt_data_fn = self._get_grad_log_p_wrt_data_fn()

        self._grad_log_p_wrt_params_fn = self._get_grad_log_p_wrt_params_fn()

        self._log_p_fn = self._get_log_p_fn()

    def get_grad_log_p_wrt_data_fn(self):
        return self._grad_log_p_wrt_data_fn

    def get_grad_log_p_wrt_params_fn(self):
        return self._get_grad_log_p_wrt_params_fn()

    def get_log_p_fn(self):
        return self._log_p_fn

    def _get_grad_log_p_wrt_data_fn(self):
        @numba.jit(nopython=True)
        def grad_fn(x, params):
            params = np.array([params[0], precision])

            return pgfa.distributions.normal.grad_log_normal_pdf_wrt_data(x, params)

        precision = self.precision

        return grad_fn

    def _get_grad_log_p_wrt_params_fn(self):
        @numba.jit(nopython=True)
        def grad_fn(x, params):
            params = np.array([params[0], precision])

            return pgfa.distributions.normal.grad_log_normal_pdf_wrt_params(x, params)

        precision = self.precision

        return grad_fn

    def _get_log_p_fn(self):
        @numba.jit(nopython=True)
        def log_p_fn(x, params):
            params = np.array([params[0], precision])

            return pgfa.distributions.normal.log_normal_pdf(x, params)

        precision = self.precision

        return log_p_fn

    def rvs(self, params, size=None):
        params = self._get_params(params)

        mean = params[0]

        std_dev = 1 / self.precision

        x = self._get_data(np.random.normal(mean, scale=std_dev, size=size))

        if size is not None:
            x = x.reshape((size, self.data_dim))

        return x


if __name__ == '__main__':
    main()
