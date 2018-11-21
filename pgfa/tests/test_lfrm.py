import unittest

import numpy as np
import scipy.special
import scipy.stats

from pgfa.math_utils import bernoulli_rvs

import pgfa.models.lfrm as lfrm
import pgfa.feature_allocation_distributions as fa


class Test(unittest.TestCase):

    def test_log_p(self):
        for _ in range(100):
            dist = lfrm.DataDistribution()

            data, params = self._simulate(4, 20)

            log_p_true = self._log_p_true(data, params)

            log_p_test = dist.log_p(data, params)

            self.assertAlmostEqual(log_p_test, log_p_true)

    def test_tau_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(4, 100)

            model = lfrm.Model(data, None, params=params.copy())

            model.params.tau_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                lfrm.update_tau(model)

                if i % num_updates == 0:
                    trace.append(model.params.tau)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.tau))

        x, _ = np.histogram(ranks, np.arange(0, 101))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_V_update(self):
        num_replicates = 10
        num_samples = 100
        num_updates = 1

        ranks = np.zeros((num_replicates, 2))

        feat_alloc_dist = fa.BetaBernoulliFeatureAllocationDistribution(4)

        for r in range(num_replicates):
            data, params = self._simulate(4, 2)

            model = lfrm.Model(data, feat_alloc_dist, params=params.copy(), symmetric=False)

            trace = np.zeros((num_samples, 2))

            idx = 0

            for i in range(num_samples * num_updates):
                lfrm.update_V(model)

                if i % num_updates == 0:
                    trace[idx, 0] = np.mean(model.params.V)

                    trace[idx, 1] = np.var(model.params.V)

                    idx += 1

            ranks[r, 0] = np.sum(trace[:, 0] < np.mean(params.V))

            ranks[r, 1] = np.sum(trace[:, 1] < np.var(params.V))

        for i in range(2):
            x, _ = np.histogram(ranks[:, i], np.arange(0, 101))

            result = scipy.stats.chisquare(x)

            self.assertGreater(result.pvalue, 1e-2)

    def _log_p_true(self, data, params):
        log_p = 0

        for i in range(params.N):
            for j in range(params.N):
                m = params.Z[i].T @ params.V @ params.Z[j]

                if data[i, j] == 0:
                    log_p += np.log(1 - scipy.special.expit(m))

                else:
                    log_p += np.log(scipy.special.expit(m))

        return log_p

    def _simulate(self, K, N, alpha_prior=np.ones(2), tau_prior=np.ones(2), feat_alloc_dist=None):
        alpha = scipy.stats.gamma.rvs(alpha_prior[0], scale=(1 / alpha_prior[1]))

        if feat_alloc_dist is None:
            Z = np.random.randint(0, 2, size=(N, K))

        else:
            Z = feat_alloc_dist.rvs(alpha, N)

        K = Z.shape[1]

        tau = scipy.stats.gamma.rvs(tau_prior[0], scale=(1 / tau_prior[1]))

        V = np.random.normal(0, 1 / np.sqrt(tau), size=(K, K))

        params = lfrm.Parameters(alpha, alpha_prior, tau, tau_prior, V, Z)

        data = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                m = Z[i].T @ V @ Z[j]

                f = np.exp(-m)

                p = 1 / (1 + f)

                data[i, j] = bernoulli_rvs(p)

        return data, params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
