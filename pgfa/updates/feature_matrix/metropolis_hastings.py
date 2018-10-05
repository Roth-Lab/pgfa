import numba
import numpy as np

from .utils import get_rows


class MetropolisHastingsUpdater(object):
    def __init__(self, adaptation_rate=None, flip_prob=0.5, target_accept_rate=0.44):
        self.adaptation_rate = adaptation_rate

        self.flip_prob = flip_prob

        self.target_accept_rate = target_accept_rate

        self._num_accepted = 0

        self._num_proposed = 0

        self._num_proposed_since_adapted = 0

    @property
    def accept_rate(self):
        return self._num_accepted / self._num_proposed

    def update(self, data, dist, feat_alloc_prior, params):
        for row_idx in get_rows(params.N):
            cols = feat_alloc_prior.get_update_cols(row_idx, params.Z)

            feat_probs = feat_alloc_prior.get_feature_probs(row_idx, params.Z)

            accept, params = do_metropolis_hastings_update(
                data, dist, cols, feat_probs, params, row_idx, flip_prob=self.flip_prob
            )

            self._num_accepted += accept

            if self.adaptation_rate is not None:
                if self._num_proposed_since_adapted >= self.adaptation_rate:
                    self._adapt()

                    self._num_proposed_since_adapted = 0

            self._num_proposed += 1

            self._num_proposed_since_adapted += 1

        return params

    def _adapt(self):
        eps = min(0.25, 1 / np.sqrt(self._num_proposed))

        if self.accept_rate > self.target_accept_rate:
            self.flip_prob = sigmoid(logit(self.flip_prob) + eps)

        elif self.accept_rate < self.target_accept_rate:
            self.flip_prob = sigmoid(logit(self.flip_prob) - eps)


def logit(x):
    return np.log(x) - np.log(1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.jit
def do_metropolis_hastings_update(data, dist, cols, feat_probs, params, row_idx, flip_prob=0.5):
    z = params.Z[row_idx]

    z_old = z[cols]

    z_new = z_old.copy()

    for k in cols:
        if np.random.random() <= flip_prob:
            z_new[k] = 1 - z_new[k]

    params.Z[row_idx] = z_old

    log_p_old = np.sum(z_old * np.log(feat_probs[cols])) + np.sum((1 - z_old) * np.log(1 - feat_probs[cols])) + \
        dist.log_p_row(data, params, row_idx)

    params.Z[row_idx] = z_new

    log_p_new = np.sum(z_new * np.log(feat_probs[cols])) + np.sum((1 - z_new) * np.log(1 - feat_probs[cols])) + \
        dist.log_p_row(data, params, row_idx)

    diff = log_p_new - log_p_old

    u = np.random.random()

    if np.log(u) <= diff:
        z[cols] = z_new

        accept = 1

    else:
        z[cols] = z_old

        accept = 0

    params.Z[row_idx] = z

    return accept, params
