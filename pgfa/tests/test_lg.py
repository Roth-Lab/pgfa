import unittest

import numpy as np
import scipy.stats

import pgfa.models.linear_gaussian as lg
import pgfa.feature_allocation_distributions as fa


class Test(unittest.TestCase):

    def test_log_p(self):
        for _ in range(100):
            dist = lg.DataDistribution()

            data, params = self._simulate(10, 4, 100)

            log_p_true = self._log_p_true(data, params)

            log_p_test = dist.log_p(data, params)

            self.assertAlmostEqual(log_p_test, log_p_true)

    def test_log_p_row(self):
        dist = lg.DataDistribution()

        for _ in range(100):
            data, params = self._simulate(10, 4, 100)

            log_p_true = self._log_p_true(data, params)

            log_p_test = 0

            for row_idx in range(params.N):
                log_p_test += dist.log_p_row(data, params, row_idx)

            self.assertAlmostEqual(log_p_test, log_p_true)

    def test_alpha_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 10

        feat_alloc_dist = fa.BetaBernoulliFeatureAllocationDistribution(4)

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100, feat_alloc_dist=feat_alloc_dist)

            model = lg.Model(data, feat_alloc_dist, params=params.copy())

            model.params.alpha_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                fa.update_alpha(model)

                if i % num_updates == 0:
                    trace.append(model.params.alpha)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.alpha))

        x, _ = np.histogram(ranks, np.arange(0, 101))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_alpha_ibp_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 10

        feat_alloc_dist = fa.IndianBuffetProcessDistribution()

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100, feat_alloc_dist=feat_alloc_dist)

            model = lg.Model(data, feat_alloc_dist, params=params.copy())

            model.params.alpha_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                fa.update_alpha(model)

                if i % num_updates == 0:
                    trace.append(model.params.alpha)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.alpha))

        x, _ = np.histogram(ranks, np.arange(0, 101))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_tau_v_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = lg.Model(data, None, params=params.copy())

            model.params.tau_v_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                lg.update_tau_v(model)

                if i % num_updates == 0:
                    trace.append(model.params.tau_v)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.tau_v))

        x, _ = np.histogram(ranks, np.arange(0, 101))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_tau_x_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = []

        for _ in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = lg.Model(data, None, params=params.copy())

            model.params.tau_x_prior = np.array([0.1, 0.1])

            trace = []

            for i in range(num_samples * num_updates):
                lg.update_tau_x(model)

                if i % num_updates == 0:
                    trace.append(model.params.tau_x)

            trace = np.array(trace)

            ranks.append(np.sum(trace < params.tau_x))

        x, _ = np.histogram(ranks, np.arange(0, 101))

        result = scipy.stats.chisquare(x)

        self.assertGreater(result.pvalue, 1e-2)

    def test_V_update(self):
        num_replicates = 100
        num_samples = 100
        num_updates = 1

        ranks = np.zeros((num_replicates, 2))

        for r in range(num_replicates):
            data, params = self._simulate(10, 4, 100)

            model = lg.Model(data, None, params=params.copy())

            trace = np.zeros((num_samples, 2))

            idx = 0

            for i in range(num_samples * num_updates):
                lg.update_V(model)

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
        return scipy.stats.matrix_normal.logpdf(
            data,
            mean=params.Z @ params.V,
            rowcov=(1 / params.tau_x) * np.eye(params.N),
            colcov=np.eye(params.D)
        )

    def _simulate(
            self,
            D,
            K,
            N,
            alpha_prior=np.ones(2),
            tau_v_prior=np.ones(2),
            tau_x_prior=np.ones(2),
            feat_alloc_dist=None):

        alpha = scipy.stats.gamma.rvs(alpha_prior[0], scale=(1 / alpha_prior[1]))

        if feat_alloc_dist is None:
            Z = np.random.randint(0, 2, size=(N, K))

        else:
            Z = feat_alloc_dist.rvs(alpha, N)

        K = Z.shape[1]

        tau_v = scipy.stats.gamma.rvs(tau_v_prior[0], scale=(1 / tau_v_prior[1]))

        tau_x = scipy.stats.gamma.rvs(tau_x_prior[0], scale=(1 / tau_x_prior[1]))

        if K == 0:
            V = np.zeros((K, D))

            data = scipy.stats.matrix_normal.rvs(
                mean=np.zeros((N, D)),
                rowcov=(1 / tau_x) * np.eye(N),
                colcov=np.eye(D)
            )

        else:
            V = scipy.stats.matrix_normal.rvs(
                mean=np.zeros((K, D)),
                rowcov=(1 / tau_v) * np.eye(K),
                colcov=np.eye(D)
            )

            data = scipy.stats.matrix_normal.rvs(
                mean=Z @ V,
                rowcov=(1 / tau_x) * np.eye(N),
                colcov=np.eye(D)
            )

        params = lg.Parameters(alpha, alpha_prior, tau_v, tau_v_prior, tau_x, tau_x_prior, V, Z)

        return data, params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
