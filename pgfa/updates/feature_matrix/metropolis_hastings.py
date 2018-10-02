import numpy as np


def do_metropolis_hastings_update(density, a, b, cols, row_idx, data, params, rho=0.5):
    z_old = params.Z[row_idx]

    z_new = z_old.copy()

    for i in range(len(z_new)):
        if np.random.random() <= rho:
            z_new[i] = 1 - z_new[i]

    params.Z[row_idx] = z_old

    log_p_old = density.log_p(data, params)

    params.Z[row_idx] = z_new

    log_p_new = density.log_p(data, params)

    diff = log_p_new - log_p_old

    u = np.random.random()

    if np.log(u) <= diff:
        params.Z[row_idx] = z_new

    else:
        params.Z[row_idx] = z_old

    return params
