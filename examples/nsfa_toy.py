import numpy as np

from pgfa.models.nsfa import NSFA, Parameters, get_min_error, get_rmse
from pgfa.utils import get_b_cubed_score


def main():
    num_iters = 10000

    params_true = get_params()

    data = get_data(params_true)

    print(data[:, 0])
    print(data[:, 100])
    print(data[:, 200])

    print()

    model = NSFA()

    params = model.get_init_params(data, 3)

    priors = model.get_priors(ibp=True)

    priors.gamma = np.array([1, 1])
    priors.U = np.array([0.1, 0.1])

#     params.gamma = params_true.gamma
#     params.F = params_true.F
#     params.S = params_true.S
#     params.U = params_true.U
#     params.V = params_true.V

    for i in range(num_iters):
        if i % 1 == 0:
            print(
                i,
                params.K,
                model.log_predictive_pdf(data, params)
            )

            print(model.log_pdf(data, params, priors))

            print(get_rmse(data, params_true), get_rmse(data, params), get_min_error(params, params_true))

            print(get_b_cubed_score(params_true.U, params.U))

            print(np.sum(np.sum(params.U, axis=0) > 0))

            print()

        model._update_gamma(params, priors)
        model._update_F(data, params)
        model._update_S(data, params, priors)
        model._update_U(data, params, priors)
#         model._update_U_row(data, params, priors)
        model._update_V(data, params)

    print(params.gamma)

    print(params.V)

    print(params.W)

    print()

    X_pred = np.dot(params.W, params.F)

    print(params.F[:, 0])
    print(params.F[:, 100])
    print(params.F[:, 200])
    print()

    print(X_pred[:, 0])
    print(X_pred[:, 100])
    print(X_pred[:, 200])


def get_data(params):
    return np.dot(params.W, params.F) + np.random.multivariate_normal(np.zeros(params.D), params.S, size=params.N).T


def get_params():
    D = 10
    K = 2
    N = 300

    gamma = 1

    F = np.zeros((K, N))
    F[0, :100] = 1
    F[1, 100:200] = 1
    F[:, 200:] = 1

    V = np.arange(1, D * K + 1).reshape((D, K))

    U = np.zeros((D, K))
    U[:5, 0] = 1
    U[5:, 1] = 1

    S = np.eye(D)

    return Parameters(gamma, F, S, U, V)


if __name__ == '__main__':
    main()
