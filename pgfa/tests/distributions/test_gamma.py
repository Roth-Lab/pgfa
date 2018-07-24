import numpy as np
import scipy.special
import scipy.stats
import sympy
import sympy.stats
import sympy.utilities
import unittest

from pgfa.distributions import GammaDistribution

from .base import TestDistributionCases


class TestGammaDistribution(TestDistributionCases.TestDistribution):
    def _eval_grad_log_pdf_wrt_data(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_data()

        shape, scale = np.squeeze(params)

        grad = grad_fn(data, shape, scale)

        grad = np.squeeze(grad)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad))

        return grad

    def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_params()

        shape, scale = np.squeeze(params)

        print(shape, scale)

        grad = grad_fn(data, shape, scale)

        grad = np.squeeze(grad)

        grad = np.swapaxes(grad, 0, 1)

        if bulk_sum:
            grad = np.atleast_2d(np.sum(grad, axis=0))

        return grad

    def _eval_log_pdf(self, data, params):
        shape, scale = params[0]

        return np.squeeze(scipy.stats.gamma.logpdf(data, shape, scale=scale))

    def _get_dist(self):
        return GammaDistribution()

    def _get_grad_test_params(self):
        params = []

        for shape in [1, 10, 100]:
            for scale in [0.1, 1, 10, 100]:
                params.append(np.atleast_2d(np.array([shape, scale])))

        return params

    def _get_sympy_grad_wrt_data(self):
        x, shape, scale = sympy.symbols('x shape scale')

        X = sympy.stats.Gamma('X', shape, scale)

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, shape, scale),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('x')),
            ]).T,
            modules=[np, scipy.special]
        )

    def _get_sympy_grad_wrt_params(self):
        x, shape, scale = sympy.symbols('x shape scale')

        X = sympy.stats.Gamma('X', shape, scale)

        log_pdf = sympy.log(sympy.stats.density(X)(x))

        return sympy.utilities.lambdify(
            (x, shape, scale),
            sympy.Matrix([
                sympy.simplify(log_pdf.diff('shape')),
                sympy.simplify(log_pdf.diff('scale'))
            ]).T,
            modules=[np, scipy.special]
        )

    def _get_test_data(self):
        data = np.linspace(0.1, 100, 1000)

        data = data.reshape((1000, 1))

        return data

    def _get_test_params(self):
        params = []

        for shape in [0.1, 1, 10, 100, 1000]:
            for scale in [0.1, 1, 10, 100, 1000]:
                params.append(np.atleast_2d(np.array([shape, scale])))

        return params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
