import numpy as np
import scipy.stats
import unittest

from pgfa.distributions import DirichletDistribution

from .base import TestDistributionCases


class TestDirichletDistribution(TestDistributionCases.TestDistribution):

    dim = 10

    def test_grad_log_p_wrt_data(self):
        data = self._get_test_data()

        params = self._get_grad_test_params()

        with self.assertRaises(NotImplementedError):
            self.dist.grad_log_p_wrt_data(data, params[0], bulk_sum=True),

    def test_grad_log_p_wrt_params(self):
        data = self._get_test_data()

        params = self._get_grad_test_params()

        with self.assertRaises(NotImplementedError):
            self.dist.grad_log_p_wrt_params(data, params[0], bulk_sum=True)

    def _eval_log_pdf(self, data, params):
        params = np.squeeze(params)

        if data.ndim == 2:
            data = np.swapaxes(data, 0, 1)

        return np.squeeze(scipy.stats.dirichlet.logpdf(data, params))

    def _get_dist(self):
        return DirichletDistribution(self.dim)

    def _get_grad_test_params(self):
        return self._get_test_params()

    def _get_test_data(self):
        return np.random.dirichlet(np.ones(self.dim), size=100)

    def _get_test_params(self):
        params = np.array([
            np.ones(self.dim) * 1e-3,
            np.ones(self.dim),
            np.ones(self.dim) * 1e3,
            np.arange(1, self.dim + 1)
        ])

        return params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
