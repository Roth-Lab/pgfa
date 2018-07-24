import numpy as np
import scipy.stats
import sympy
import sympy.stats
import sympy.utilities
import unittest

from pgfa.distributions import NormalDistribution

from .base import TestDistributionCases


class TestNormalDistribution(TestDistributionCases.TestDistribution):
    def _eval_grad_log_pdf_wrt_data(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_data()

        mean, precision = np.squeeze(params)

        grad = grad_fn(data, mean, precision)

        grad = np.squeeze(grad)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad))

        return grad

    def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_params()

        mean, precision = np.squeeze(params)

        grad = grad_fn(data, mean, precision)

        grad = np.squeeze(grad)

        grad = np.swapaxes(grad, 0, 1)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad, axis=0))

        return grad

    def _eval_log_pdf(self, data, params):
        mean, precision = params[0]

        std_dev = 1 / np.sqrt(precision)

        return np.squeeze(scipy.stats.norm.logpdf(data, loc=mean, scale=std_dev))

    def _get_dist(self):
        return NormalDistribution()

    def _get_grad_test_params(self):
        return self._get_test_params()

    def _get_sympy_grad_wrt_data(self):
        x, m, p = sympy.symbols('x m p')

        X = sympy.stats.Normal('X', m, 1 / sympy.sqrt(p))

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, m, p),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('x')),
            ]).T
        )

    def _get_sympy_grad_wrt_params(self):
        x, m, p = sympy.symbols('x m p')

        X = sympy.stats.Normal('X', m, 1 / sympy.sqrt(p))

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, m, p),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('m')),
                sympy.simplify(log_pdf.diff('p'))
            ]).T
        )

    def _get_test_data(self):
        data = np.linspace(-100, 100, 1000)

        data = data.reshape((1000, 1))

        return data

    def _get_test_params(self):
        params = []

        for mean in [-100, -10, -1, 0, 1, 10, 100]:
            for precision in [0.1, 1, 10, 100]:
                params.append(np.atleast_2d(np.array([mean, precision])))

        return params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
