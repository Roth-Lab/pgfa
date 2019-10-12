import unittest

from collections import defaultdict, Counter

import numpy as np

from pgfa.feature_allocation_distributions import BetaBernoulliFeatureAllocationDistribution
from pgfa.updates import DiscreteParticleFilterUpdater, GibbsUpdater, ParticleGibbsUpdater, RowGibbsUpdater

from pgfa.tests.exact_posterior import get_exact_posterior
from pgfa.tests.mocks import MockDataDistribution, MockParams


class Test(unittest.TestCase):
    K = 5
    N = 3
    D = 10

    def test_discrete_particle_filter_updater(self):
        feat_alloc_updater = DiscreteParticleFilterUpdater(annealing_power=1.0, num_particles=10)

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_discrete_particle_filter_annealed_updater(self):
        feat_alloc_updater = DiscreteParticleFilterUpdater(annealing_power=1.0, num_particles=10)

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_gibbs(self):
        feat_alloc_updater = GibbsUpdater()

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_particle_gibbs_updater(self):
        feat_alloc_updater = ParticleGibbsUpdater(annealing_power=0.0, num_particles=10, resample_threshold=0.5)

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_particle_gibbs_annealed_updater(self):
        feat_alloc_updater = ParticleGibbsUpdater(annealing_power=1.0, num_particles=10, resample_threshold=0.5)

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_restricted_row_gibbs(self):
        feat_alloc_updater = RowGibbsUpdater(max_cols=3)

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def test_row_gibbs(self):
        feat_alloc_updater = RowGibbsUpdater()

        model = self._get_model()

        self._run_test(feat_alloc_updater, model, num_iters=int(1e4))

    def _get_model(self):
        data = np.random.normal(0, 1, size=(self.N, self.D))

        dist = MockDataDistribution()

        feat_alloc_dist = BetaBernoulliFeatureAllocationDistribution(self.K)

        params = MockParams(2, self.K, self.N)

        return TestModel(data, dist, feat_alloc_dist, params)

    def _get_pred_posterior(self, feat_alloc_updater, model, burnin=int(1e3), num_iters=int(1e3)):
        trace = Counter()

        for _ in range(burnin):
            feat_alloc_updater.update(model)

        for _ in range(num_iters):
            feat_alloc_updater.update(model)

            key = tuple(model.params.Z.flatten())

            trace[key] += 1

        posterior = defaultdict(float)

        for key in trace:
            posterior[key] = trace[key] / num_iters

        return posterior

    def _get_true_posterior(self, model):
        return get_exact_posterior(model)

    def _run_test(self, feat_alloc_updater, model, burnin=int(1e3), num_iters=int(1e3)):
        pred_posterior = self._get_pred_posterior(feat_alloc_updater, model, burnin=burnin, num_iters=num_iters)

        true_posterior = self._get_true_posterior(model)

        self._test_posterior(pred_posterior, true_posterior)

    def _test_posterior(self, pred_probs, true_probs):
#         print(sorted(pred_probs.items()))
#         print()
#         print(sorted(true_probs.items()))
#         print()
        for key in true_probs:
            if abs(pred_probs[key] - true_probs[key]) >= 0.02:
                print(key)

            self.assertAlmostEqual(pred_probs[key], true_probs[key], delta=0.02)


class TestModel(object):

    def __init__(self, data, dist, feat_alloc_dist, params):
        self.data = data

        self.data_dist = dist

        self.feat_alloc_dist = feat_alloc_dist

        self.params = params

    @property
    def log_p(self):
        log_p = 0

        log_p += self.data_dist.log_p(self.data, self.params)

        log_p += self.feat_alloc_dist.log_p(self.params)

        return log_p


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_gibbs']
    unittest.main()
