import unittest

from collections import defaultdict, Counter

import numpy as np

from pgfa.feature_allocation_priors import BetaBernoulliFeatureAllocationDistribution
from pgfa.updates.feature_matrix import GibbsUpdater, ParticleGibbsUpdater, RowGibbsUpdater

from pgfa.tests.exact_posterior import get_exact_posterior
from pgfa.tests.mocks import MockDistribution, MockFeatureAllocationPrior, MockParams


class Test(unittest.TestCase):
    K = 4
    N = 3
    D = 10

    def test_gibbs(self):
        feat_alloc_updater = GibbsUpdater()

        model = self._get_model(feat_alloc_updater)

        self._run_test(model, num_iters=int(1e4))

    def test_particle_gibbs_updater(self):
        feat_alloc_updater = ParticleGibbsUpdater(annealed=False, num_particles=20, resample_threshold=0.5)

        model = self._get_model(feat_alloc_updater)

        self._run_test(model, num_iters=int(1e4))

    def test_particle_gibbs_annealed_updater(self):
        feat_alloc_updater = ParticleGibbsUpdater(annealed=True, num_particles=20, resample_threshold=0.5)

        model = self._get_model(feat_alloc_updater)

        self._run_test(model, num_iters=int(1e4))

    def test_restricted_row_gibbs(self):
        feat_alloc_updater = RowGibbsUpdater(max_cols=3)

        model = self._get_model(feat_alloc_updater)

        self._run_test(model, num_iters=int(1e4))

    def test_row_gibbs(self):
        feat_alloc_updater = RowGibbsUpdater()

        model = self._get_model(feat_alloc_updater)

        self._run_test(model, num_iters=int(1e4))

    def _get_model(self, feat_alloc_updater):
        data = np.random.normal(0, 1, size=(self.N, self.D))

        dist = MockDistribution()

        feat_alloc_prior = BetaBernoulliFeatureAllocationDistribution(1, 1, self.K)

        params = MockParams(self.K, self.N)

        return TestModel(data, dist, feat_alloc_prior, feat_alloc_updater, params)

    def _get_pred_posterior(self, model, burnin=int(1e3), num_iters=int(1e3)):
        trace = Counter()

        for _ in range(burnin):
            model.update()

        for _ in range(num_iters):
            model.update()

            key = tuple(model.params.Z.flatten())

            trace[key] += 1

        posterior = defaultdict(float)

        for key in trace:
            posterior[key] = trace[key] / num_iters

        return posterior

    def _get_true_posterior(self, model):
        return get_exact_posterior(model)

    def _run_test(self, model, burnin=int(1e3), num_iters=int(1e3)):
        pred_posterior = self._get_pred_posterior(model, burnin=burnin, num_iters=num_iters)

        true_posterior = self._get_true_posterior(model)

        self._test_posterior(pred_posterior, true_posterior)

    def _test_posterior(self, pred_probs, true_probs):
        print(sorted(pred_probs.items()))
        print()
        print(sorted(true_probs.items()))
        print()
        for key in true_probs:
            if abs(pred_probs[key] - true_probs[key]) >= 0.02:
                print(key)

            self.assertAlmostEqual(pred_probs[key], true_probs[key], delta=0.02)


class TestModel(object):
    def __init__(self, data, dist, feat_alloc_prior, feat_alloc_updater, params):
        self.data = data

        self.dist = dist

        self.feat_alloc_prior = feat_alloc_prior

        self.feat_alloc_updater = feat_alloc_updater

        self.params = params

    @property
    def log_p(self):
        log_p = 0

        log_p += self.dist.log_p(self.data, self.params)

        log_p += self.feat_alloc_prior.log_p(self.params.Z)

        return log_p

    def update(self):
        self.params = self.feat_alloc_updater.update(self.data, self.dist, self.feat_alloc_prior, self.params)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_gibbs']
    unittest.main()
