import numba
import numpy as np

from pgfa.math_utils import get_linear_sum_params, log_normalize
from pgfa.stats import discrete_rvs


class GibbsFeatureAllocationMatrixKernel(object):
    """ Kernel to update entries of the feature matrix conditional on all others.
    """

    def __init__(self, do_ibp_update=False):
        self.do_ibp_update = do_ibp_update

    def update(self, model):
        """ Update all (instantiated) entries of the feature allocation matrix.

        Note: This class makes a copy of the feature allocation matrix so the model is unchanged.

        Parameters
        ----------
        model: FeatureAllocationModel
            Model to compute update feature allocation matrix.

        Returns
        -------
        feat_mat: ndarray
            Update feature allocation matrix.
        """
        feat_mat = model.latent_values.copy()

        log_p_fn = model.feature_dist.get_log_p_fn()

        if self.do_ibp_update:
            feat_mat = _update_feature_allocation_matrix_ibp(
                log_p_fn,
                model.feature_weight_params,
                model.feature_params,
                model.data,
                feat_mat
            )

        else:
            feat_mat = _update_feature_allocation_matrix(
                log_p_fn,
                model.feature_weight_params,
                model.feature_params,
                model.data,
                feat_mat
            )

        return feat_mat


@numba.jit(nopython=True)
def _update_feature_allocation_matrix(log_p_fn, rho_priors, theta, X, Z):
    """ Outer loop over rows for finite feature allocation model.
    """
    K = Z.shape[1]

    N = Z.shape[0]

    rows = np.arange(N)

    np.random.shuffle(rows)

    cols = np.arange(K)

    np.random.shuffle(cols)

    for n in rows:
        m = np.sum(Z, axis=0)

        m -= Z[n]

        a = m + rho_priors[0]

        b = (N - 1 - m) + rho_priors[1]

        z = Z[n]

        z = _update_row(
            log_p_fn,
            a[cols],
            b[cols],
            theta[cols],
            X[n],
            z[cols]
        )

        for k in range(K):
            Z[n, cols[k]] = z[k]

    return Z


@numba.jit(nopython=True)
def _update_feature_allocation_matrix_ibp(log_p_fn, theta, X, Z):
    """ Outer loop over rows for the Indian Buffet Process model.
    """
    K = Z.shape[1]

    N = Z.shape[0]

    rows = np.arange(N)

    np.random.shuffle(rows)

    cols = np.arange(K)

    np.random.shuffle(cols)

    for n in rows:
        m = np.sum(Z, axis=0)

        m -= Z[n]

        a = m

        b = (N - 1 - m)

        z = Z[n]

        z = _update_row(
            log_p_fn,
            a[cols],
            b[cols],
            theta[cols],
            X[n],
            z[cols]
        )

        for k in range(K):
            Z[n, cols[k]] = z[k]

    return Z


@numba.jit(nopython=True)
def _update_row(log_p_fn, a, b, theta, x, z):
    """ Update a single row of the feature allocation matrix.
    """
    K = len(z)

    log_p = np.zeros(2)

    for k in range(K):
        z[k] = 0

        f = get_linear_sum_params(z, theta)

        log_p[0] = np.log(b[k]) + log_p_fn(x, f)

        z[k] = 1

        f = get_linear_sum_params(z, theta)

        log_p[1] = np.log(a[k]) + log_p_fn(x, f)

        log_p = log_normalize(log_p)

        z[k] = discrete_rvs(np.exp(log_p))

    return z
