import numpy as np
import scipy.stats

from pgfa.math_utils import discrete_rvs, do_metropolis_hastings_accept_reject, log_normalize


class PriorSingletonsUpdater(object):

    def update_row(self, model, row_idx):
        alpha = model.params.alpha

        D = model.params.D
        N = model.params.N

        k_old = len(self._get_singleton_idxs(model.params.Z, row_idx))

        k_new = scipy.stats.poisson.rvs(alpha / N)

        if (k_new == 0) and (k_old == 0):
            return model.params

        non_singleton_idxs = self._get_non_singleton_idxs(model.params.Z, row_idx)

        num_non_singletons = len(non_singleton_idxs)

        K_new = len(non_singleton_idxs) + k_new

        params_old = model.params.copy()

        params_new = model.params.copy()

        params_new.V = np.zeros((K_new, D))

        params_new.V[:num_non_singletons] = model.params.V[non_singleton_idxs]

        if k_new > 0:
            a, b = model.params.V_prior

            params_new.V[num_non_singletons:] = scipy.stats.gamma.rvs(a, scale=(1 / b), size=(k_new, model.params.D))

        params_new.Z = np.zeros((N, K_new), dtype=np.int8)

        params_new.Z[:, :num_non_singletons] = model.params.Z[:, non_singleton_idxs]

        params_new.Z[row_idx, num_non_singletons:] = 1

        log_p_new = model.data_dist.log_p(model.data, params_new)

        log_p_old = model.data_dist.log_p(model.data, model.params)

        if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
            model.params = params_new

        else:
            model.params = params_old

    def _get_column_counts(self, Z, row_idx):
        m = np.sum(Z, axis=0)

        m -= Z[row_idx]

        return m

    def _get_non_singleton_idxs(self, Z, row_idx):
        m = self._get_column_counts(Z, row_idx)

        return np.atleast_1d(np.squeeze(np.where(m > 0)))

    def _get_singleton_idxs(self, Z, row_idx):
        m = self._get_column_counts(Z, row_idx)

        return np.atleast_1d(np.squeeze(np.where(m == 0)))


class SplitMergeUpdater(object):

    def __init__(self, annealing_factor=1):
        self.annealing_factor = annealing_factor

    def update(self, model):
        anchors = np.random.choice(model.params.N, replace=False, size=2)

        if (np.sum(model.params.Z[anchors[0]]) == 0) or ((np.sum(model.params.Z[anchors[1]]) == 0)):
            return

        features, log_q_feature_fwd = self._select_features(anchors, model.params.Z)

        if features[0] == features[1]:
            V_new, Z_new, log_q_sm_fwd = self._propose_split(anchors, features, model, model.params.V, model.params.Z)

            _, log_q_feature_rev = self._select_features(anchors, Z_new)

            K_new = Z_new.shape[1]

            _, _, log_q_sm_rev = self._propose_merge(anchors, np.array([K_new - 2, K_new - 1]), V_new, Z_new)

        else:
            V_new, Z_new, log_q_sm_fwd = self._propose_merge(anchors, features, model.params.V, model.params.Z)

            _, log_q_feature_rev = self._select_features(anchors, Z_new)

            K_new = Z_new.shape[1]

            _, _, log_q_sm_rev = self._propose_split(anchors, np.array([K_new - 1, K_new - 1]), model, V_new, Z_new, Z_target=model.params.Z[:, features])

        params_fwd = model.params.copy()

        params_fwd.V = V_new

        params_fwd.Z = Z_new

        log_p_fwd = model.joint_dist.log_p(model.data, params_fwd)

        params_rev = model.params

        log_p_rev = model.joint_dist.log_p(model.data, params_rev)

        if do_metropolis_hastings_accept_reject(log_p_fwd, log_p_rev, self.annealing_factor * (log_q_feature_fwd + log_q_sm_fwd), self.annealing_factor * (log_q_feature_rev + log_q_sm_rev)):
            model.params = params_fwd

        else:
            model.params = params_rev

    def _propose_merge(self, anchors, features, V, Z):
        k_a, k_b = features

        _, D = V.shape

        N, K = Z.shape

        V_new = np.zeros((K - 1, D), dtype=V.dtype)

        Z_new = np.zeros((N, K - 1), dtype=Z.dtype)

        idx = 0

        for k in range(K):
            if k in features:
                continue

            V_new[idx] = V[k]

            Z_new[:, idx] = Z[:, k]

            idx += 1

        active_set = list(set(np.atleast_1d(np.squeeze(np.where(Z[:, k_a] == 1)))) | set(np.atleast_1d(np.squeeze(np.where(Z[:, k_b] == 1)))))

        Z_new[active_set, -1] = 1

        V_new[-1] = V[k_a] + V[k_b]

        return V_new, Z_new, 0

    def _propose_split(self, anchors, features, model, V, Z, Z_target=None):
        k_m = features[0]

        i, j = anchors

        _, D = V.shape

        N, K = Z.shape

        V_new = np.zeros((K + 1, D), dtype=V.dtype)

        Z_new = np.zeros((N, K + 1), dtype=Z.dtype)

        idx = 0

        for k in range(K):
            if k in features:
                continue

            V_new[idx] = V[k]

            Z_new[:, idx] = Z[:, k]

            idx += 1

        weight = np.random.random(D)

        V_new[-1] = weight * V[k_m]

        V_new[-2] = (1 - weight) * V[k_m]

        Z_new[i, -1] = 1

        Z_new[j, -2] = 1

        active_set = list(np.squeeze(np.where(Z[:, k_m] == 1)))

        active_set.remove(i)

        active_set.remove(j)

        np.random.shuffle(active_set)

        log_q = 0

        log_p = np.zeros(3)

        params = model.params.copy()

        params.V = V_new

        params.Z = Z_new

        N_prev = 2

        for idx in active_set:  # + [i, j]:
            if idx not in [i, j]:
                N_prev += 1

            m_a = np.sum(Z_new[:, -1])

            m_b = np.sum(Z_new[:, -2])

            params.Z[idx, -1] = 1

            params.Z[idx, -2] = 0

            log_p[0] = np.log(m_a) + np.log(N_prev - m_b) + model.data_dist.log_p_row(model.data, params, idx)

            params.Z[idx, -1] = 0

            params.Z[idx, -2] = 1

            log_p[1] = np.log(N_prev - m_a) + np.log(m_b) + model.data_dist.log_p_row(model.data, params, idx)

            params.Z[idx, -1] = 1

            params.Z[idx, -2] = 1

            log_p[2] = np.log(m_a) + np.log(m_b) + model.data_dist.log_p_row(model.data, params, idx)

            log_p = log_normalize(log_p)

            if Z_target is None:
                state = discrete_rvs(np.exp(log_p))

            else:
                if np.all(Z_target[idx] == np.array([1, 0])):
                    state = 0

                elif np.all(Z_target[idx] == np.array([0, 1])):
                    state = 1

                elif np.all(Z_target[idx] == np.array([1, 1])):
                    state = 2

                else:
                    raise Exception('Invalid')

            if state == 0:
                Z_new[idx, -1] = 1

                Z_new[idx, -2] = 0

            elif state == 1:
                Z_new[idx, -1] = 0

                Z_new[idx, -2] = 1

            elif state == 2:
                Z_new[idx, -1] = 1

                Z_new[idx, -2] = 1

            else:
                raise Exception('Invalid state')

            log_q += log_p[state]

        assert Z_new is params.Z

        return V_new, Z_new, log_q

    def _select_features(self, anchors, Z):
        i, j = anchors

        log_q = 0

        k_a = np.random.choice(np.where(Z[i] == 1)[0])

        log_q -= np.log(np.sum(Z[i]))

        k_b = np.random.choice(np.where(Z[j] == 1)[0])

        log_q -= np.log(np.sum(Z[j]))

        return np.array([k_a, k_b]), log_q
