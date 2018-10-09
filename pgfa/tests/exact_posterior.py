import itertools
import numpy as np

from pgfa.math_utils import log_normalize


def get_all_binary_matrices(K, N):
    for Z in itertools.product([0, 1], repeat=N * K):
        Z = np.array(Z).reshape((N, K)).astype(int)

        yield Z


def get_exact_posterior(model):
    log_p = []

    Zs = []

    for Z in get_all_binary_matrices(model.params.K, model.data.shape[0]):
        Zs.append(tuple(Z.flatten()))

        model.params.Z = Z

        log_p.append(model.log_p)

    p = np.exp(log_normalize(np.array(log_p)))

    return dict(list(zip(Zs, p)))
