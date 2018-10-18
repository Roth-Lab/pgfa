import numba
import numpy as np

from pgfa.math_utils import discrete_rvs, log_beta, log_factorial


class BetaBernoulliFeatureAllocationDistribution(object):
    def __init__(self, a, b, K):
        self.a = a

        self.b = b

        self.K = K

    def get_feature_probs(self, row_idx, Z):
        N = Z.shape[0]

        m = _get_conditional_counts(row_idx, Z)

        a = self.a + m

        b = self.b + (N - 1 - m)

        return a / (a + b)

    def get_update_cols(self, row_idx, Z):
        assert Z.shape[1] == self.K

        cols = np.arange(self.K)

        np.random.shuffle(cols)

        return cols

    def log_p(self, Z):
        assert Z.shape[1] == self.K

        K = Z.shape[1]

        N = Z.shape[0]

        if K == 0:
            return 0

        m = np.sum(Z, axis=0)

        a0 = self.a

        b0 = self.b

        a = a0 + m

        b = b0 + (N - m)

        return np.sum(log_beta(a, b) - log_beta(a0, b0))

    def rvs(self, N):
        K = self.K

        p = np.random.beta(self.a, self.b, size=K)

        Z = np.zeros((N, K), dtype=np.int64)

        for k in range(K):
            Z[:, k] = np.random.multinomial(1, [1 - p[k], p[k]], size=N).argmax(axis=1)

        return Z

    def sample_num_singletons(self, Z):
        raise NotImplementedError

    def update(self, Z):
        pass


class IndianBuffetProcessDistribution(object):
    def __init__(self, alpha=1, priors=np.array([1, 1])):
        self.alpha = alpha

        self.priors = priors

    def get_feature_probs(self, row_idx, Z):
        N = Z.shape[0]

        m = _get_conditional_counts(row_idx, Z)

        return m / N

    def get_update_cols(self, row_idx, Z):
        m = _get_conditional_counts(row_idx, Z)

        cols = [k for k in range(Z.shape[1]) if (m[k] > 0)]

        np.random.shuffle(cols)

        return cols

    def log_p(self, Z):
        alpha = self.alpha

        K = Z.shape[1]

        N = Z.shape[0]
        if K == 0:
            return 0

        H = np.sum(1 / np.arange(1, N + 1))

        histories, history_counts = np.unique(Z, axis=1, return_counts=True)

        m = histories.sum(axis=0)

        num_histories = histories.shape[1]

        log_p = K * np.log(alpha) - H * alpha

        for h in range(num_histories):
            K_h = history_counts[h]

            log_p -= log_factorial(K_h)

            log_p += K_h * log_factorial(m[h] - 1) + K_h * log_factorial(N - m[h])

            log_p -= history_counts[h] * log_factorial(N)

        return log_p

    def rvs(self, N):
        K = np.random.poisson(self.alpha)

        Z = np.ones((1, K), dtype=np.int64)

        for n in range(1, N):
            K = Z.shape[1]

            z = np.zeros(K)

            m = np.sum(Z, axis=0)

            for k in range(K):
                p = np.array([n - m[k], m[k]])

                p = p / p.sum()

                z[k] = discrete_rvs(p)

            Z = np.row_stack([Z, z])

            k_new = np.random.poisson(self.alpha / (n + 1))

            if k_new > 0:
                Z = np.column_stack([Z, np.zeros((Z.shape[0], k_new))])

                Z[n, K:] = 1

        return Z.astype(np.int64)

    def sample_num_singletons(self, Z):
        N = Z.shape[0]

        return np.random.poisson(self.alpha / N)

    def update(self, Z):
        K = Z.shape[1]

        N = Z.shape[0]

        a = K + self.priors[0]

        b = np.sum(1 / np.arange(1, N + 1)) + self.priors[1]

        self.alpha = np.random.gamma(a, 1 / b)


@numba.njit(cache=True)
def _get_conditional_counts(row_idx, Z):
    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return m
