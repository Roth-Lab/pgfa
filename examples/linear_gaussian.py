import numpy as np
import os
import scipy.stats

import pgfa.models.linear_gaussian
import pgfa.math_utils
import pgfa.utils

os.environ['NUMBA_WARNINGS'] = '1'


def main():
    np.random.seed(0)

    num_iters = 10001
    D = 10
    K = 4
    N = 100

    data, _, Z = simulate_data(3, D, N, K=K, s_a=0.01, s_x=10)

    init_params = pgfa.models.linear_gaussian.get_params_from_data(data, K=1)

    print(np.sum(Z, axis=0))

    params = init_params.copy()

#     model = pgfa.models.linear_gaussian.LinearGaussianModel(data, K=None, params=params)

    model = pgfa.models.linear_gaussian.CollapsedLinearGaussianModel(data, K=None, params=params)

    update = 'pg'

    print(update)

    for i in range(num_iters):
        if i % 10 == 0:
            print(
                i,
                model.params.Z.shape[1],
                model.params.alpha,
                model.params.tau_a,
                model.params.tau_x,
                #                 model.log_p,
            )

            print(pgfa.utils.get_b_cubed_score(Z, model.params.Z))

            print((1 / (N * D)) * np.sqrt(np.sum((data - model.params.Z @ model.params.V)**2)))

            print(np.sum(model.params.Z, axis=0))

        model.update(update_type=update)


def simulate_data(alpha, D, N, K=None, s_a=1, s_x=1):
    if K is None:
        Z = pgfa.math_utils.ibp_rvs(alpha, N)
        K = Z.shape[1]

    else:
        Z = pgfa.math_utils.ffa_rvs(alpha / K, 1, K, N)

    A = scipy.stats.matrix_normal.rvs(
        mean=np.zeros((K, D)),
        rowcov=(1 / s_a) * np.eye(K),
        colcov=np.eye(D)
    )

    data = scipy.stats.matrix_normal.rvs(
        mean=np.dot(Z, A),
        rowcov=(1 / s_x) * np.eye(N),
        colcov=np.eye(D)
    )

    return data, A, Z


if __name__ == '__main__':
    main()
