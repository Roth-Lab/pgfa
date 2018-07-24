import numba
import numpy as np
import scipy.stats


@numba.jit(nopython=True)
def discrete_rvs(p):
    """ Simulate a Bernoulli random variable.

    Parameters
    ----------
    p: ndarray
        Array of class probabilites. Must sum to 1.

    Returns:
    X: int
        Discrete value selected.
    """
    p = p + 1e-10

    p = p / np.sum(p)

    return np.random.multinomial(1, p).argmax()


def gamma_rvs(shape, scale, size=None):
    """ Simulate a Gamma random variable.

    Definition of parameters matches Wikipedia.

    Parameters
    ----------
    shape: float
        Shape parameter of Gamma distribution.
    scale: float
        Scale parameter of Gamma distribution.
    size: scalar or arraylike
        Size of array to simulate.
    """
    return scipy.stats.gamma.rvs(shape, scale=scale, size=size)
