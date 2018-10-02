import numpy as np

from pgfa.math_utils import ibp_rvs, ffa_rvs
from pgfa.models.pyclone import Parameters, PyCloneFeatureAllocationModel
from pgfa.utils import get_b_cubed_score


def main():
    np.random.seed(0)

    D = 4
    K = 5
    N = 100

    update_type = 'pga'

    do_gibbs = True

    if do_gibbs:
        print(update_type + ' + gibbs')

    else:
        print(update_type)

    data, params = simulate_data(D, N, K=K)

    print(np.sum(params.Z, axis=0))

    model = PyCloneFeatureAllocationModel(data, K=K)

    model.params.eta = params.eta.copy()

    model.params.Z = np.random.randint(0, 2, size=params.Z.shape)

    for i in range(10000):
        if i % 100 == 0:
            print(i, model.params.K, model.params.alpha, model.log_p)
            print(np.sum(model.params.Z, axis=0))
            print(get_b_cubed_score(params.Z, model.params.Z))
            print('#' * 100)

        model.update(num_particles=1000, resample_threshold=0.5, update_type=update_type)

        if do_gibbs:
            model.update(update_type='g')


def simulate_data(D, N, alpha=1, kappa=10, K=None):
    if K is None:
        Z = ibp_rvs(alpha, N)

    else:
        Z = ffa_rvs(1, 1, K, N)

    K = Z.shape[1]

    eta = np.random.gamma(kappa, 1, size=(D, K))

    phi = eta / np.sum(eta, axis=1)[:, np.newaxis]

    data = np.zeros((N, D, 2))

    for d in range(D):
        for n in range(N):
            data[n, d, 0] = np.random.poisson(10000)

            data[n, d, 1] = np.random.binomial(data[n, d, 0], np.sum(Z[n] * phi[d]))

    params = Parameters(alpha, kappa, eta, Z)

    return data, params


if __name__ == '__main__':
    main()
