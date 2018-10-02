import numpy as np
import os
import scipy.stats

import pgfa.models.nsfa
import pgfa.math_utils
import pgfa.utils
from pgfa.math_utils import ffa_rvs, ibp_rvs

os.environ['NUMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 100001
    D = 10
    K = 4
    N_train = 1000
    N_test = 1000

    params_train = get_test_params(N_train, K, D, seed=0)

    params_test = get_test_params(N_test, K, D, params=params_train, seed=1)

    data_train = get_data(params_train)

    data_test = get_data(params_test)

    print(params_train.Z.sum(axis=0))

    print(params_train.F.min(), params_train.F.max())

    update = 'g'

    print(update)

    print(params_train.Z.shape)

    print(pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModel(data_train, K=K, params=params_train).log_p)

    print('@' * 100)

    model = pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModel(data_train, K=None)

    old_params = model.params

    model.params = params_train.copy()

    model.params.Z = np.random.randint(0, 2, size=(model.params.D, model.params.K))

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
                print(model.params.Z)
                print(pgfa.utils.get_b_cubed_score(params_train.Z, model.params.Z))

            print(np.sum(model.params.Z, axis=0))

            print(model.params.S)


#             print(pgfa.models.nsfa.get_min_error(model.params, params_train))
#
#             print(get_min_error(model.params, params_train))

#             print(model.params.V[0])

            print('#' * 100)

        model.update(update_type=update, num_particles=20)


def get_data(params):
    data = params.W @ params.F

    data += np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T

    return data


def get_test_params(num_data_points, num_latent_dims, num_observed_dims, params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    D = num_observed_dims
    K = num_latent_dims
    N = num_data_points

    alpha = 2

    if K is None:
        if params is None:
            Z = ibp_rvs(alpha, D)

        else:
            Z = params.Z.copy()

        K = Z.shape[1]

    else:
        Z = ffa_rvs(1, 1, K, D)

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

    return pgfa.models.nsfa.Parameters(alpha, gamma, F, S, V, Z)


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
