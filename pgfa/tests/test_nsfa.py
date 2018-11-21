import unittest

import numpy as np
import scipy.stats

import pgfa.models.nsfa as nsfa
import pgfa.feature_allocation_distributions as fa


class Test(unittest.TestCase):

    def test_log_p(self):
        for _ in range(100):
            dist = nsfa.DataDistribution()

            data, params = self._simulate(10, 4, 100)

            log_p_true = self._log_p_true(data, params)

            log_p_test = dist.log_p(data, params)

            self.assertAlmostEqual(log_p_test, log_p_true)

    def test_log_p_row(self):
        dist = nsfa.DataDistribution()

        for _ in range(100):
            data, params = self._simulate(10, 4, 100)

            log_p_true = self._log_p_true(data, params)

            log_p_test = 0

            for row_idx in range(params.D):
                log_p_test += dist.log_p_row(data, params, row_idx)

            self.assertAlmostEqual(log_p_test, log_p_true)

    def test_gamma_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = nsfa.Model(data, None, params=params.copy())

            model.params.gamma_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                nsfa.update_gamma(model)

                if i % num_updates == 0:
                    trace.append(model.params.gamma)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.gamma))

        x, _ = np.histogram(ranks, np.arange(0, num_samples + 1))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_S_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = nsfa.Model(data, None, params=params.copy())

            model.params.S_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                nsfa.update_S(model)

                if i % num_updates == 0:
                    trace.append(model.params.S)

            trace = np.array(trace)

            ranks.append([np.sum(trace[:, d] < params.S[d]) for d in range(params.D)])

        ranks = np.array(ranks)

        for d in range(params.D):
            x, _ = np.histogram(ranks[:, d], np.arange(0, num_samples + 1))

            result = scipy.stats.chisquare(x)

            self.assertGreater(result.pvalue, 1e-2)

    def test_F_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = nsfa.Model(data, None, params=params.copy())

            trace = []

            for i in range(num_samples * num_updates):
                nsfa.update_F(model)

                if i % num_updates == 0:
                    trace.append([np.mean(model.params.F), np.var(model.params.F)])

            trace = np.array(trace)

            ranks.append([
                np.sum(trace[:, 0] < np.mean(params.F)),
                np.sum(trace[:, 1] < np.var(params.F))
            ])

        ranks = np.array(ranks)

        for i in range(2):
            x, _ = np.histogram(ranks[:, i], np.arange(0, num_samples + 1))

            result = scipy.stats.chisquare(x)

            self.assertGreater(result.pvalue, 1e-2)

    def test_V_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = nsfa.Model(data, None, params=params.copy())

            trace = []

            for i in range(num_samples * num_updates):
                nsfa.update_V(model)

                if i % num_updates == 0:
                    trace.append([np.mean(model.params.V), np.var(model.params.V)])

            trace = np.array(trace)

            ranks.append([
                np.sum(trace[:, 0] < np.mean(params.V)),
                np.sum(trace[:, 1] < np.var(params.V))
            ])

        ranks = np.array(ranks)

        for i in range(2):
            x, _ = np.histogram(ranks[:, i], np.arange(0, num_samples + 1))

            result = scipy.stats.chisquare(x)

            self.assertGreater(result.pvalue, 1e-2)

    def _log_p_true(self, data, params):
        log_p = 0

        for n in range(params.N):
            log_p += scipy.stats.multivariate_normal.logpdf(
                data[:, n],
                params.W @ params.F[:, n],
                np.diag(1 / params.S)
            )

        return log_p

    def _simulate(
            self,
            D,
            K,
            N,
            alpha_prior=np.ones(2),
            gamma_prior=np.ones(2),
            S_prior=np.ones(2),
            feat_alloc_dist=None):

        alpha = scipy.stats.gamma.rvs(alpha_prior[0], scale=(1 / alpha_prior[1]))

        if feat_alloc_dist is None:
            Z = np.random.randint(0, 2, size=(D, K))

        else:
            Z = feat_alloc_dist.rvs(alpha, D)

        K = Z.shape[1]

        gamma = scipy.stats.gamma.rvs(gamma_prior[0], scale=(1 / gamma_prior[1]))

        S = scipy.stats.gamma.rvs(S_prior[0], scale=(1 / S_prior[1]), size=D)

        V = scipy.stats.multivariate_normal.rvs(np.zeros(K), (1 / gamma) * np.eye(K), size=D)

        F = scipy.stats.norm.rvs(0, 1, size=(K, N))

        params = nsfa.Parameters(alpha, alpha_prior, gamma, gamma_prior, F, S, S_prior, V, Z)

        data = params.W @ params.F

        data += np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T

        return data, params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
