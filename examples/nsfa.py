import numpy as np

from pgfa.models.nsfa import NSFA, Parameters, get_min_error, get_rmse
from pgfa.utils import get_b_cubed_score


def main():
    num_iters = 1000
    D = 10
    K = 2
    N_train = 100
    N_test = 100

    params_train = get_test_params(N_train, K, D, seed=0)

    params_test = get_test_params(N_test, K, D, params=params_train, seed=1)

    data_train = get_data(params_train)

    data_test = get_data(params_test)

    model = NSFA()

    params = model.get_init_params(data_train, K, seed=2)

    priors = model.get_priors()

    priors.U = np.array([1, 1])

    for i in range(num_iters):
        if i % 100 == 0:
            print(
                i,
                model.log_predictive_pdf(data_train, params),
                model.log_predictive_pdf(data_test, params)
            )

            print(get_rmse(data_train, params_train), get_rmse(data_train, params), get_min_error(params, params_train))

            print(get_b_cubed_score(params_train.U, params.U))

            print(np.sum(np.sum(params.U, axis=0) > 0))

            print()

        params = model.update_params(data_train, params, priors)

        params.F = params_train.F
        params.S = params_train.S
#         params.U = params_train.U
        params.V = params_train.V


def get_data(params):
    return np.dot(params.W, params.F) + np.random.multivariate_normal(np.zeros(params.D), params.S, size=params.N).T


def get_test_params(num_data_points, num_latent_dims, num_observed_dims, params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    D = num_observed_dims
    K = num_latent_dims
    N = num_data_points

    F = np.random.normal(0, 1, size=(K, N))

    if params is None:
        gamma = 100

#         S = np.diag(np.random.gamma(1, 10, size=D))
        S = np.diagflat(np.ones(D) * 10)

        U = np.random.randint(0, 2, size=(D, K))

        V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=D)

#         V = np.random.multivariate_normal(np.zeros(K), gamma * np.eye(K), size=D)
#         V = np.linspace(-10, 10, num=D * K)
#
#         np.random.shuffle(V)
#
#         V = V.reshape((D, K))

    else:
        gamma = params.gamma

        S = params.S.copy()

        U = params.U.copy()

        V = params.V.copy()

    return Parameters(gamma, F, S, U, V)


if __name__ == '__main__':
    main()
