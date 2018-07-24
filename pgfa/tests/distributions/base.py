import numpy as np
import unittest


class TestDistributionCases(object):

    class TestDistribution(unittest.TestCase):

        def _eval_grad_log_pdf_wrt_data(self, data, params, bulk_sum=True):
            raise NotImplementedError()

        def _eval_grad_log_pdf_wrt_params(self, data, params, bulk_sum=True):
            raise NotImplementedError()

        def _eval_log_pdf(self, data, params):
            raise NotImplementedError()

        def _get_dist(self):
            raise NotImplementedError()

        def _get_grad_test_params(self):
            raise NotImplementedError()

        def _get_test_data(self):
            raise NotImplementedError()

        def _get_test_params(self):
            raise NotImplementedError()

        def setUp(self):
            self.dist = self._get_dist()

        def test_grad_log_p_wrt_data(self):
            try:
                data = self._get_test_data()

                params = self._get_grad_test_params()

                for p in params:
                    np.testing.assert_allclose(
                        self.dist.grad_log_p_wrt_data(data, p, bulk_sum=True),
                        self._eval_grad_log_pdf_wrt_data(data, p, bulk_sum=True),
                        atol=1e-3,
                        rtol=1e-4
                    )

            except NotImplementedError:
                pass

        def test_grad_log_p_wrt_params(self):
            data = self._get_test_data()

            params = self._get_grad_test_params()

            for p in params:
                np.testing.assert_allclose(
                    self.dist.grad_log_p_wrt_params(data, p, bulk_sum=True),
                    self._eval_grad_log_pdf_wrt_params(data, p, bulk_sum=True),
                    atol=1e-4,
                    rtol=1e-4
                )

        def test_log_p_bulk(self):
            data = self._get_test_data()

            params = self._get_test_params()

            for p in params:
                np.testing.assert_allclose(
                    self.dist.log_p(data, p, bulk_sum=False),
                    np.squeeze(self._eval_log_pdf(data, p))
                )

        def test_log_p_bulk_sum(self):
            data = self._get_test_data()

            params = self._get_test_params()

            for p in params:
                np.testing.assert_allclose(
                    self.dist.log_p(data, p, bulk_sum=True),
                    np.sum(self._eval_log_pdf(data, p))
                )

        def test_log_p(self):
            data = self._get_test_data()

            params = self._get_test_params()

            N = data.shape[0]

            for p in params:
                for n in range(N):
                    self.assertAlmostEqual(
                        self.dist.log_p(data[n], p),
                        self._eval_log_pdf(data[n], p)
                    )

        def test_rvs(self):
            params = self._get_test_params()[0]

            self.assertTupleEqual(self.dist.rvs(params, size=100).shape, (100, self.dist.data_dim))
