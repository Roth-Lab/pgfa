import numpy as np
import os

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution
from pgfa.models.nsfa import Parameters, NonparametricSparaseFactorAnalysisModel
from pgfa.updates.feature_matrix import GibbsMixtureUpdater, ParticleGibbsUpdater
from pgfa.utils import get_b_cubed_score

os.environ['NUMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 100001
    D = 10
    K = 4
    N_train = 1000
    N_test = 1000

    feat_alloc_prior = BetaBernoulliFeatureAllocationDistribution(1, 1, K)

    params_train = get_test_params(feat_alloc_prior, N_train, D, seed=0)

    params_test = get_test_params(feat_alloc_prior, N_test, D, params=params_train, seed=1)

    data_train = get_data(params_train)

    data_test = get_data(params_test)

    feat_alloc_updater = ParticleGibbsUpdater(annealed=True)

    feat_alloc_updater = GibbsMixtureUpdater(feat_alloc_updater)

    model = NonparametricSparaseFactorAnalysisModel(data_train, feat_alloc_prior, feat_alloc_updater)

    for i in range(num_iters):
        if i % 100 == 0:
            print(
                i,
                model.params.Z.shape[1],
                model.params.alpha,
                model.params.gamma,
                model.log_p,
                model.log_predictive_pdf(data_test),
                model.rmse,
            )

            if model.params.K > 0:
                print(get_b_cubed_score(params_train.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model.update()


def get_data(params):
    data = params.W @ params.F

    data += np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T

    return data


def get_test_params(feat_alloc_prior, num_data_points, num_observed_dims, params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    D = num_observed_dims
    N = num_data_points

    alpha = 2

    Z = feat_alloc_prior.rvs(D)

    K = Z.shape[1]

    F = np.random.normal(0, 1, size=(K, N))

    if params is None:
        gamma = 0.01

        S = 10 * np.ones(D)

        V = np.random.multivariate_normal(np.zeros(K), (1 / gamma) * np.eye(K), size=D)

    else:
        gamma = params.gamma

        S = params.S.copy()

        Z = params.Z.copy()

        V = params.V.copy()

    return Parameters(alpha, gamma, F, S, V, Z)


def get_min_error(params_pred, params_true):
    import itertools

    W_pred = params_pred.F.T

    W_true = params_true.F.T

    K_pred = W_pred.shape[1]

    K_true = W_true.shape[1]

    min_error = float('inf')

    for perm in itertools.permutations(range(K_pred)):
        error = np.sqrt(np.mean((W_pred[:, perm[:K_true]] - W_true)**2))

        if error < min_error:
            min_error = error

    return min_error


if __name__ == '__main__':
    main()
