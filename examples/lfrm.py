import numpy as np

from pgfa.math_utils import bernoulli_rvs, ffa_rvs, ibp_rvs
from pgfa.models.lfrm import LatentFactorRelationalModel, Parameters
from pgfa.utils import get_b_cubed_score


def main():
    np.random.seed(0)

    K = 4
    N = 20

    update_type = 'g'

    print(update_type)

    data, data_true, params = simulate_data(N, K=K)

    print(np.sum(params.Z, axis=0))

    print(params.V)

    print(LatentFactorRelationalModel(data, K=K, params=params).log_p)

    print('#' * 100)

    model = LatentFactorRelationalModel(data, K=None)

    model.params = params.copy()

#     model.params.tau = 1e-6

#     model.params.Z = params.Z.copy()

#     model.params.V = params.V.copy()

    model.priors.Z = np.array([0.1, 0.9])

    for i in range(10000):
        if i % 100 == 0:
            print(i, model.params.K, model.params.alpha, model.params.tau, model.log_p)
            print(np.sum(model.params.Z, axis=0))
            print(get_b_cubed_score(params.Z, model.params.Z))
            print(1 / (params.N ** 2) * np.sum(np.abs(data_true - model.predict(method='max'))))
            if model.params.K <= 4:
                print(model.params.V)
            print('#' * 100)

        model.update(update_type=update_type)


def simulate_data(N, K=None, alpha=1, tau=1):
    if K is None:
        Z = ibp_rvs(alpha, N)

        K = Z.shape[1]

    else:
        Z = ffa_rvs(1, 1, K, N)

#     V = np.random.normal(0, 1 / np.sqrt(tau), size=K)
#     V = np.random.gamma(10, 10, size=K)
#     V = np.diag(V)

    V = np.random.normal(0, 1 / np.sqrt(tau), size=(K, K))
#     V = np.random.gamma(10, 10, size=(K, K))
    V = np.triu(V)
    V = V + V.T - np.diag(np.diag(V))

    params = Parameters(alpha, tau, V, Z)

    data_true = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                data_true[i, j] = 1

            else:
                m = Z[i].T @ V @ Z[j]

                f = np.exp(-m)

                p = 1 / (1 + f)

                data_true[i, j] = bernoulli_rvs(p)

    data = data_true.copy()

    for i in range(N):
        for j in range(N):
            if np.random.random() < 0.1:
                data[i, j] = np.nan

    return data, data_true, params


if __name__ == '__main__':
    main()
