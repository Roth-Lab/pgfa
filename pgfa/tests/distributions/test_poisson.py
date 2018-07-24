import numpy as np
import scipy.stats
import sympy
import sympy.stats
import sympy.utilities
import unittest

from pgfa.distributions import PoissonDistribution

from .base import TestDistributionCases


class TestPoissonDistribution(TestDistributionCases.TestDistribution):
    def _eval_log_pdf(self, data, params):
        rate = params[0][0]

        return np.squeeze(scipy.stats.poisson.logpmf(data, rate))

    def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_params()

        rate = np.squeeze(params)

        grad = grad_fn(data, rate)

        grad = np.squeeze(grad)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad, axis=0))

        return grad

    def _get_dist(self):
        return PoissonDistribution()

    def _get_grad_test_params(self):
        return self._get_test_params()

    def _get_sympy_grad_wrt_params(self):
        x, rate = sympy.symbols('x rate')

        X = sympy.stats.Poisson('X', rate)

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, rate),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('rate'))
            ]).T
        )

    def _get_test_data(self):
        data = np.arange(0, 1000)

        data = data.reshape((1000, 1))

        return data

    def _get_test_params(self):
        params = []

        for rate in [0.1, 1, 10, 100, 1000]:
            params.append(np.atleast_2d(np.array([rate])))

        return params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
