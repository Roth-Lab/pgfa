import numpy as np

from pgfa.models.nsfa import NonparametricSparaseFactorAnalysisModel, Parameters

import pgfa.math_utils


def main():
    D = 2
    N = 2

    for _ in range(100):
        params = simulate_params(D, N)

        data = simulate_data(params)

        trace = fit_model(data)

        print(np.mean(trace, axis=0))


def fit_model(data, burnin=1000, num_iters=1000, thin=10):
    model = NonparametricSparaseFactorAnalysisModel(data)

    trace = []

    for i in range(-burnin, num_iters):
        model.update(update_type='g')

        if (i > 0) and (i % thin == 0):
            trace.append([model.params.alpha, model.params.gamma, model.params.S[0], model.params.K])

    return np.array(trace)


def simulate_params(D, N):
    alpha = 2

    while True:
        Z = pgfa.math_utils.ibp_rvs(alpha, D)

        if Z.shape[1] > 0:
            break

    K = Z.shape[1]

    gamma = 0.1

    F = np.random.normal(0, 1, size=(K, N))

    S = 10 * np.ones(D)

    V = np.random.multivariate_normal(np.zeros(K), (1 / gamma) * np.eye(K), size=D)

    params = Parameters(alpha, gamma, F, S, V, Z)

    return params


def simulate_data(params):
    eps = np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T

    return params.W @ params.F + eps


if __name__ == '__main__':
    main()
