import numpy as np

from pgfa.distributions import GammaDistribution, NormalDistribution
from pgfa.mcmc_utils import do_metropolis_hastings_accept_reject


class MetropolisHastingsFeatureKernel(object):
    def __init__(self, data_dist, proposal_dist, adaptation_rate=None):
        self.data_dist = data_dist

        self.proposal_dist = proposal_dist

        self.adaptation_rate = adaptation_rate

        self._num_accepted = 0

        self._num_proposed = 0

        self._num_proposed_since_adapted = 0

    @property
    def accept_rate(self):
        return self._num_accepted / self._num_proposed

    def sample(self, data, feat_mat, feat_params):
        D = feat_params.shape[1]

        Ds = np.arange(D)

        np.random.shuffle(Ds)

        K = feat_params.shape[0]

        Ks = np.arange(K)

        np.random.shuffle(Ks)

        for k in Ks:
            for d in Ds:
                f_old = feat_params[k, d]

                f_new = self.proposal_dist.sample(f_old)

                feat_params[k, d] = f_new

                log_p_new = self.data_dist.log_p(data, feat_mat, feat_params)

                feat_params[k, d] = f_old

                log_p_old = self.data_dist.log_p(data, feat_mat, feat_params)

                log_q_new = self.proposal_dist.log_p(f_new, f_old)

                log_q_old = self.proposal_dist.log_p(f_old, f_new)

                if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, log_q_new, log_q_old):
                    feat_params[k, d] = f_new

                    self._num_accepted += 1

                else:
                    feat_params[k, d] = f_old

                self._num_proposed += 1

                self._num_proposed_since_adapted = 0

            if self.adaptation_rate is not None:
                if self._num_proposed_since_adapted >= self.adaptation_rate:
                    self.proposal_dist.adapt(self.accept_rate, self._num_proposed)

                    self._num_proposed_since_adapted = 0

        return feat_params


class GammaRandomWalkProposal(object):
    """ Random walk proposal from a Gamma distribution centred at current value.

    Parameters
    ----------
    precision: float
        Precision of Gamma distribution to propose from. Larger values will lead to smaller moves, but higher
        accpetance rates.
    """

    def __init__(self, precision=1):
        self.precision = precision

        self.dist = GammaDistribution()

    def adapt(self, accept_rate, num_iters, target=0.44):
        eps = min(0.25, 1 / np.sqrt(num_iters))

        if accept_rate > target:
            self.precision = np.exp(np.log(self.precision) - eps)

        elif accept_rate < target:
            self.precision = np.exp(np.log(self.precision) + eps)

    def log_p(self, data, params):
        params = self._get_standard_params(params)

        return self.dist.log_p(data, params)

    def sample(self, params):
        params = self._get_standard_params(params)

        return self.dist.rvs(params)

    def _get_standard_params(self, params):
        scale = self.precision * params

        shape = params / scale

        return np.array([shape, scale])


class NormalRandomWalkProposal(object):
    """ Random walk proposal from a Normal distribution centred at current value.

    Parameters
    ----------
    precision: float
        Precision of Gamma distribution to propose from. Larger values will lead to smaller moves, but higher
        accpetance rates.
    """

    def __init__(self, precision=1):
        self.precision = precision

        self.dist = NormalDistribution()

    def adapt(self, accept_rate, num_iters, target=0.44):
        eps = min(0.25, 1 / np.sqrt(num_iters))

        if accept_rate > target:
            self.precision = np.exp(np.log(self.precision) - eps)

        elif accept_rate < target:
            self.precision = np.exp(np.log(self.precision) + eps)

    def log_p(self, data, params):
        params = self._get_standard_params(params)

        return self.dist.log_p(data, params)

    def sample(self, params):
        params = self._get_standard_params(params)

        return self.dist.rvs(params)

    def _get_standard_params(self, params):
        mean = np.squeeze(params)

        precision = self.precision

        return np.array([mean, precision])
