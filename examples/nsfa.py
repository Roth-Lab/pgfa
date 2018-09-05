import numpy as np
import os
import scipy.stats

import pgfa.models.nsfa
import pgfa.math_utils
import pgfa.utils
from pgfa.math_utils import ffa_rvs

os.environ['NZMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 10001
    D = 10
    K = 4
    N_train = 100
    N_test = 100

    params_train = get_test_params(N_train, K, D, seed=0)

    params_test = get_test_params(N_test, K, D, params=params_train, seed=1)

    data_train = get_data(params_train)

    data_test = get_data(params_test)

    print(data_train.shape)

    params = pgfa.models.nsfa.get_params_from_data(data_train, K=1)

    model = pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModel(data_train, K=None, params=params)

    for i in range(num_iters):
        if i % 10 == 0:
            print(
                i,
                model.params.Z.shape[1],
                model.params.alpha,
                model.log_p,
                model.log_predictive_pdf(data_test)
            )

            print(pgfa.utils.get_b_cubed_score(params_train.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

        model.update(update_type='pga')


def get_data(params):
    print(np.dot(params.W, params.F).shape)
    return np.dot(params.W, params.F) + \
        np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T


def get_test_params(num_data_points, num_latent_dims, num_observed_dims, params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    D = num_observed_dims
    K = num_latent_dims
    N = num_data_points

    alpha = 1

    F = np.random.normal(0, 1, size=(K, N))

    if params is None:
        gamma = 100

        S = np.random.gamma(1, 10, size=D)
#         S = np.diagflat(np.ones(D) * 10)

        Z = np.random.randint(0, 2, size=(D, K))

        Z = ffa_rvs(1, 1, K, D)

#         Z = np.ones((D, K))

        V = np.random.multivariate_normal(np.zeros(K), 10 * np.eye(K), size=D).T

#         V = np.random.multivariate_normal(np.zeros(K), gamma * np.eye(K), size=D)
#         V = np.linspace(-10, 10, num=D * K)
#
#         np.random.shuffle(V)

        V = V.reshape((K, D))

    else:
        gamma = params.gamma

        S = params.S.copy()

        Z = params.Z.copy()

        V = params.V.copy()

    return pgfa.models.nsfa.Parameters(alpha, gamma, F, S, V, Z)


if __name__ == '__main__':
    main()
