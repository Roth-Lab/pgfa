import itertools
import numba
import numpy as np

from pgfa.math_utils import get_linear_sum_params, log_normalize
from pgfa.stats import discrete_rvs


class RowGibbsFeatureAllocationMatrixKernel(object):
    """ Kernel to update entries of the feature matrix by row conditioned on other rows.

    Note: This kernel scales as O(2**K) so it will be inefficient.
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

        # Enumerate all possible binary vectors of length K
        K = feat_mat.shape[1]

        Zs = list(map(np.array, itertools.product([0, 1], repeat=K)))

        Zs = np.array(Zs, dtype=np.int)

        if self.do_ibp_update:
            feat_mat = _update_feature_allocation_matrix_ibp(
                log_p_fn,
                model.feature_params,
                model.data,
                feat_mat,
                Zs
            )

        else:
            feat_mat = _update_feature_allocation_matrix(
                log_p_fn,
                model.feature_weight_params,
                model.feature_params,
                model.data,
                feat_mat,
                Zs
            )

        return feat_mat


@numba.jit(nopython=True)
def _update_feature_allocation_matrix(log_p_fn, rho_priors, theta, X, Z, Zs):
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
            z[cols],
            Zs
        )

        for k in range(K):
            Z[n, cols[k]] = z[k]

    return Z


@numba.jit(nopython=True)
def _update_feature_allocation_matrix_ibp(log_p_fn, theta, X, Z, Zs):
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
            z[cols],
            Zs
        )

        for k in range(K):
            Z[n, cols[k]] = z[k]

    return Z


@numba.jit(nopython=True)
def _update_row(log_p_fn, a, b, theta, x, z, Zs):
    M = len(Zs)

    log_p = np.zeros(M)

    for m in range(M):
        z = Zs[m]

        f = get_linear_sum_params(z, theta)

        log_p[m] = np.sum(z * np.log(a)) + np.sum((1 - z) * np.log(b)) + log_p_fn(x, f)

    log_p = log_normalize(log_p)

    p = np.exp(log_p)

    idx = discrete_rvs(p)

    return Zs[idx]
