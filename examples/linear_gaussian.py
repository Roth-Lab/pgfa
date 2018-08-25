import numpy as np
import scipy.stats

import pgfa.models.linear_gaussian
import pgfa.stats
import pgfa.utils


def main():
    np.random.seed(0)

    D = 10
    K = 4
    N = 100

    data, A, Z = simulate_data(20, D, N, K=K, s_a=0.1, s_x=10)

    params = pgfa.models.linear_gaussian.get_params(D, N, 1)

    model = pgfa.models.linear_gaussian.LinearGaussianCollapsed(data, params=params)

    for i in range(100):
        if i % 10 == 0:
            print(
                i,
                model.params.Z.shape[1],
                model.params.tau_a,
                model.params.tau_x,
                model.get_log_p_X(data, model.params.Z),
                model.get_log_p_X_collapsed(data, model.params.Z)
            )

            print(pgfa.utils.get_b_cubed_score(Z, model.params.Z))

        model.update(kernel='gibbs')


def simulate_data(alpha, D, N, K=None, s_a=1, s_x=1):
    if K is None:
        Z = pgfa.stats.ibp_rvs(alpha, N)
        K = Z.shape[1]

    else:
        Z = pgfa.stats.ffa_rvs(alpha / K, 1, K, N)

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
