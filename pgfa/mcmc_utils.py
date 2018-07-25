import numpy as np


def do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
    u = np.random.random()

    diff = (log_p_new - log_q_new) - (log_p_old - log_q_old)

    if diff >= np.log(u):
        accept = True

    else:
        accept = False

    return accept
