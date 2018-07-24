import numpy as np
import scipy.special
import scipy.stats
import sympy
import sympy.stats
import sympy.utilities
import unittest

from pgfa.distributions import BetaDistribution

from .base import TestDistributionCases


class TestBetaDistribution(TestDistributionCases.TestDistribution):
    def _eval_grad_log_pdf_wrt_data(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_data()

        a, b = np.squeeze(params)

        grad = grad_fn(data, a, b)

        grad = np.squeeze(grad)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad))

        return grad

    def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_params()

        a, b = np.squeeze(params)

        grad = grad_fn(data, a, b)

        grad = np.squeeze(grad)

        grad = np.swapaxes(grad, 0, 1)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad, axis=0))

        return grad

    def _eval_log_pdf(self, data, params):
        a, b = np.squeeze(params)

        return np.squeeze(scipy.stats.beta.logpdf(data, a, b))

    def _get_dist(self):
        return BetaDistribution()

    def _get_grad_test_params(self):
        return self._get_test_params()

    def _get_sympy_grad_wrt_data(self):
        x, a, b = sympy.symbols('x a b')

        X = sympy.stats.Beta('X', a, b)

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, a, b),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('x')),
            ]).T,
            modules=[np, scipy.special]
        )

    def _get_sympy_grad_wrt_params(self):
        x, a, b = sympy.symbols('x a b')

        X = sympy.stats.Beta('X', a, b)

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, a, b),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('a')),
                sympy.simplify(log_pdf.diff('b'))
            ]).T,
            modules=[np, scipy.special]
        )

    def _get_test_data(self):
        eps = 1e-6

        data = np.linspace(eps, 1 - eps, 1000)

        data = data.reshape((1000, 1))

        return data

    def _get_test_params(self):
        return np.array([
            [1e-6, 1e-6],
            [1e-6, 1],
            [1, 1e-6],
            [1, 1],
            [10, 10],
            [100, 100]
        ])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
