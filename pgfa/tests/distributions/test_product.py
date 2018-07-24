import numpy as np
import scipy.stats
import sympy
import sympy.stats
import sympy.utilities
import unittest

from pgfa.distributions import NormalDistribution, ProductDistribution

from .base import TestDistributionCases

np.random.seed(0)


class TestProductDistribution(TestDistributionCases.TestDistribution):
    dim = 3

    def _eval_grad_log_pdf_wrt_data(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_data()

        data, params = self.dist._get_standard_data_params(data, params)

        grad = np.zeros((data.shape[1], data.shape[0]))

        for k in range(self.dim):
            mean, precision = np.squeeze(params[k])

            grad[:, k] = np.squeeze(grad_fn(data[k], mean, precision))

        if bulk_sum:
            grad = np.sum(grad, axis=0)

        return np.atleast_2d(grad)

    def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
        grad_fn = self._get_sympy_grad_wrt_params()

        data, params = self.dist._get_standard_data_params(data, params)

        grad = np.zeros((params.shape[0], params.shape[2], data.shape[1]))

        for k in range(self.dim):
            mean, precision = np.squeeze(params[k])

            grad[k] = np.squeeze(grad_fn(data[k], mean, precision))

        grad = np.swapaxes(grad, 0, 2)

        grad = np.swapaxes(grad, 1, 2)

        if bulk_sum:
            grad = np.sum(grad, axis=0)

            grad = grad.reshape((1, params.shape[0] * params.shape[2]))

        else:
            grad = grad.reshape((data.shape[1], params.shape[0] * params.shape[2]))

        return np.atleast_2d(grad)

    def _eval_log_pdf(self, data, params):
        data = np.atleast_2d(data)

        if data.ndim == 3:
            data = data.reshape((data.shape[0], data.shape[2]))

        params = params.reshape((self.dim, 2))

        mean = params[:, 0]

        precision = params[:, 1]

        std_dev = 1 / np.sqrt(precision)

        return np.squeeze(np.sum(scipy.stats.norm.logpdf(data, loc=mean, scale=std_dev), axis=1))

    def _get_dist(self):
        return ProductDistribution(NormalDistribution(), self.dim)

    def _get_data_dim(self):
        return 1 * self.dim

    def _get_params_dim(self):
        return 2 * self.dim

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
        data = np.random.normal(0, 100, size=(100, self.dim))

        return data

    def _get_test_params(self):
        params = np.zeros((100, self.dim, 2))

        for i in range(params.shape[0]):
            params[i, :, 0] = np.random.normal(0, 100, size=self.dim)

            params[i, :, 1] = np.random.gamma(1, 10, size=self.dim)

        params = params.reshape((100, 2 * self.dim))

        return params


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
