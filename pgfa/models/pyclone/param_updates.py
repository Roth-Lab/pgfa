import numpy as np
import scipy.stats

from pgfa.math_utils import discrete_rvs, do_metropolis_hastings_accept_reject, log_normalize, log_sum_exp


def update_precision(model, variance=1):
    old = model.params.precision

    a_new, b_new = get_gamma_params(old, variance)

    new = scipy.stats.gamma.rvs(a_new, scale=(1 / b_new))

    a_old, b_old = get_gamma_params(new, variance)

    model.params.precision = new

    log_p_new = model.joint_dist.log_p(model.data, model.params)

    log_q_new = scipy.stats.gamma.logpdf(new, a_new, scale=(1 / b_new))

    model.params.precision = old

    log_p_old = model.joint_dist.log_p(model.data, model.params)

    log_q_old = scipy.stats.gamma.logpdf(old, a_old, scale=(1 / b_old))

    if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
        model.params.precision = new

    else:
        model.params.precision = old


def update_V(model, variance=1):
    params = model.params.copy()

    a_prior, b_prior = model.params.V_prior

    Ds = np.random.permutation(model.params.D)

    Ks = np.random.permutation(model.params.K)

    for d in Ds:
        for k in Ks:
            old = params.V[k, d]

            a, b = get_gamma_params(old, variance)

            new = scipy.stats.gamma.rvs(a, scale=(1 / b))

            params.V[k, d] = new

            log_p_new = model.data_dist.log_p(model.data, params)

            log_p_new += scipy.stats.gamma.logpdf(new, a_prior, scale=(1 / b_prior))

            log_q_new = scipy.stats.gamma.logpdf(new, a, scale=(1 / b))

            a, b = get_gamma_params(new, variance)

            params.V[k, d] = old

            log_p_old = model.data_dist.log_p(model.data, params)

            log_p_old += scipy.stats.gamma.logpdf(old, a_prior, scale=(1 / b_prior))

            log_q_old = scipy.stats.gamma.logpdf(old, a, scale=(1 / b))

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                params.V[k, d] = new

            else:
                params.V[k, d] = old

    model.params = params


def update_V_perm(model):
    params = model.params.copy()

    for d in np.random.permutation(model.params.D):
        old = params.V[:, d].copy()

        new = params.V[np.random.permutation(params.K), d]

        params.V[:, d] = new

        log_p_new = model.data_dist.log_p(model.data, params)

        params.V[:, d] = old

        log_p_old = model.data_dist.log_p(model.data, params)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            params.V[:, d] = new

        else:
            params.V[:, d] = old

    model.params = params


def update_V_random_grid_pairwise(model, num_points=10):
    if model.params.K < 2:
        return

    ka, kb = np.random.choice(model.params.K, 2, replace=False)

    params = model.params.copy()

    old = params.V[[ka, kb]].flatten()

    D = params.D

    dim = 2 * D

    e = scipy.stats.multivariate_normal.rvs(np.zeros(dim), np.eye(dim))

    e /= np.linalg.norm(e)

    r = scipy.stats.gamma.rvs(1, 1)

    grid = np.arange(1, num_points + 1)

    ys = old[np.newaxis, :] + grid[:, np.newaxis] * r * e[np.newaxis, :]

    log_p_new = np.zeros(num_points)

    for i in range(num_points):
        params.V[[ka, kb]] = ys[i].reshape((2, D))

        log_p_new[i] = model.joint_dist.log_p(model.data, params)

    if np.all(np.isneginf(log_p_new)) or np.any(np.isnan(log_p_new)):
        return
    
    try:
        idx = discrete_rvs(np.exp(0.5 * np.log(grid) + log_normalize(log_p_new)))
    
    except ValueError:
        return

    new = ys[idx]

    xs = new[np.newaxis, :] - grid[:, np.newaxis] * r * e[np.newaxis, :]

    log_p_old = np.zeros(num_points)

    for i in range(num_points):
        params.V[[ka, kb]] = xs[i].reshape((2, D))

        log_p_old[i] = model.joint_dist.log_p(model.data, params)

    if do_metropolis_hastings_accept_reject(log_sum_exp(log_p_new), log_sum_exp(log_p_old), 0, 0):
        params.V[[ka, kb]] = new.reshape((2, D))

    else:
        params.V[[ka, kb]] = old.reshape((2, D))

    model.params = params


def update_V_random_grid(model, num_points=10):
    if model.params.K < 2:
        return

    params = model.params.copy()

    old = params.V.flatten()

    K, D = params.V.shape

    dim = K * D

    e = scipy.stats.multivariate_normal.rvs(np.zeros(dim), np.eye(dim))

    e /= np.linalg.norm(e)

    r = scipy.stats.gamma.rvs(1, 1)

    grid = np.arange(1, num_points + 1)

    ys = old[np.newaxis, :] + grid[:, np.newaxis] * r * e[np.newaxis, :]

    log_p_new = np.zeros(num_points)

    for i in range(num_points):
        params.V = ys[i].reshape((K, D))

        log_p_new[i] = model.joint_dist.log_p(model.data, params)

    idx = discrete_rvs(np.exp(0.5 * np.log(grid) + log_normalize(log_p_new)))

    new = ys[idx]

    xs = new[np.newaxis, :] - grid[:, np.newaxis] * r * e[np.newaxis, :]

    log_p_old = np.zeros(num_points)

    for i in range(num_points):
        params.V = xs[i].reshape((K, D))

        log_p_old[i] = model.joint_dist.log_p(model.data, params)

    if do_metropolis_hastings_accept_reject(log_sum_exp(log_p_new), log_sum_exp(log_p_old), 0, 0):
        params.V = new.reshape((K, D))

    else:
        params.V = old.reshape((K, D))

    model.params = params


def update_V_block(model, variance=1):
    params = model.params.copy()

    a_prior, b_prior = model.params.V_prior

    for k in np.random.permutation(model.params.K):
        old = params.V[k].copy()

        new = np.zeros(params.D)

        log_p_new = 0

        log_q_new = 0

        log_p_old = 0

        log_q_old = 0

        for d in range(model.params.D):
            a, b = get_gamma_params(old[d], variance)

            new[d] = scipy.stats.gamma.rvs(a, scale=(1 / b))

            log_p_new += scipy.stats.gamma.logpdf(new[d], a_prior, scale=(1 / b_prior))

            log_q_new += scipy.stats.gamma.logpdf(new[d], a, scale=(1 / b))

            a, b = get_gamma_params(new[d], variance)

            log_p_old += scipy.stats.gamma.logpdf(old[d], a_prior, scale=(1 / b_prior))

            log_q_old += scipy.stats.gamma.logpdf(old[d], a, scale=(1 / b))

        params.V[k] = new

        log_p_new += model.data_dist.log_p(model.data, params)

        params.V[k] = old

        log_p_old += model.data_dist.log_p(model.data, params)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.V[k] = new

        else:
            params.V[k] = old

    model.params = params


def update_V_block_dim(model, variance=1):
    params = model.params.copy()

    a_prior, b_prior = model.params.V_prior

    for d in np.random.permutation(model.params.D):
        old = params.V[:, d].copy()

        new = np.zeros(params.K)

        log_p_new = 0

        log_q_new = 0

        log_p_old = 0

        log_q_old = 0

        for k in range(model.params.K):
            a, b = get_gamma_params(old[k], variance)

            new[k] = scipy.stats.gamma.rvs(a, scale=(1 / b))

            log_p_new += scipy.stats.gamma.logpdf(new[k], a_prior, scale=(1 / b_prior))

            log_q_new += scipy.stats.gamma.logpdf(new[k], a, scale=(1 / b))

            a, b = get_gamma_params(new[k], variance)

            log_p_old += scipy.stats.gamma.logpdf(old[k], a_prior, scale=(1 / b_prior))

            log_q_old += scipy.stats.gamma.logpdf(old[k], a, scale=(1 / b))

        params.V[:, d] = new

        log_p_new += model.data_dist.log_p(model.data, params)

        params.V[:, d] = old

        log_p_old += model.data_dist.log_p(model.data, params)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
            params.V[:, d] = new

        else:
            params.V[:, d] = old

    model.params = params


def get_gamma_params(mean, variance):
    b = mean / variance

    a = b * mean

    return a, b
