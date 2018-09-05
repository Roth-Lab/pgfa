import numpy as np

from pgfa.math_utils import do_metropolis_hastings_accept_reject


def do_mh_singletons_update(density, proposal, data, params):
    """ Update the singletons using a Metropolis Hastings proposal from the prior.
    """
    # Current state
    log_p_old = density.log_p(data, params)

    # New state
    num_new_singletons = np.random.poisson(params.alpha / params.N)

    params_new = proposal.rvs(data, params, num_new_singletons)

    log_p_new = density.log_p(data, params_new)

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
        params = params_new

    return params

# def do_mh_singletons_update(row, density, proposal, alpha, V, X, Z):
#     """ Update the singletons using a Metropolis Hastings proposal from the prior.
#     """
#     D = V.shape[1]
#     N = Z.shape[0]
#
#     m = np.sum(Z, axis=0)
#
#     m -= Z[row]
#
#     non_singletons_idxs = np.atleast_1d(np.squeeze(np.where(m > 0)))
#
#     K_non_singleton = len(non_singletons_idxs)
#
#     # Current state
#     log_p_old = density.log_p(X[row], Z[row], V)
#
#     # New state
#     num_new_singletons = np.random.poisson(alpha / N)
#
#     K_new = K_non_singleton + num_new_singletons
#
#     z_new = np.ones(K_new)
#
#     z_new[:K_non_singleton] = Z[row, non_singletons_idxs]
#
#     V_new = np.zeros((K_new, D))
#
#     V_new[:K_non_singleton] = V[non_singletons_idxs]
#
#     V_new[K_non_singleton:] = proposal.rvs(size=num_new_singletons)
#
#     log_p_new = density.log_p(X[row], z_new, V_new)
#
#     if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
#         V = V_new
#
#         Z_new = np.zeros((N, K_new))
#
#         Z_new[:, :K_non_singleton] = Z[:, non_singletons_idxs]
#
#         Z_new[row, K_non_singleton:] = 1
#
#         Z = Z_new
#
#     return V, Z
