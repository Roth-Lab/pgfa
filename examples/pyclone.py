import numba
import numpy as np
import scipy.stats

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone.binomial
import pgfa.models.pyclone.singletons_updates
import pgfa.models.pyclone.utils
import pgfa.updates

from pgfa.utils import Timer


def main():
    seed = 0

    set_seed(seed)

    ibp = False
    time = 10000
    D = 10
    K = 8
    N = 200

    updater = 'g'
    test_path = 'conditional'

    data, params = simulate_data(D, N, K=K, alpha=2)

    model_updater = get_model_updater(
        feat_alloc_updater_type=updater, ibp=ibp, mixture_prob=0.0, test_path=test_path
    )

    sm_updater = pgfa.models.pyclone.singletons_updates.SplitMergeUpdater()

    model = get_model(data, ibp=ibp, K=K)

    print(np.sum(params.Z, axis=0))

    print('@' * 100)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    model.params.alpha = 1

    print(log_p_true)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 1 == 0:
            print(
                i,
                model.params.K,
                model.log_p,
                (model.log_p - log_p_true) / abs(log_p_true)
            )

            print(
                get_b_cubed_score(params.Z, model.params.Z)
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params)
            )

            print('#' * 100)

        timer.start()

        model_updater.update(model)

        if ibp:
            for _ in range(20):
                sm_updater.update(model)

        timer.stop()

        i += 1


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()

    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.pyclone.binomial.Model(data, feat_alloc_dist)


def get_model_updater(
        feat_alloc_updater_type='g',
        annealing_power=0,
        ibp=True,
        mixture_prob=0.0,
        num_particles=20,
        test_path='conditional'
    ):

    if ibp:
        singletons_updater = pgfa.models.pyclone.singletons_updates.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DiscreteParticleFilterUpdater(
            annealing_power=annealing_power,
            max_particles=num_particles,
            singletons_updater=singletons_updater,
            test_path=test_path
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    elif feat_alloc_updater_type == 'gpf':
        feat_alloc_updater = pgfa.updates.GumbelParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles
            )

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealing_power=annealing_power,
            num_particles=num_particles,
            singletons_updater=singletons_updater,
            test_path=test_path
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixture_prob > 0:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater, gibbs_prob=mixture_prob)

    return pgfa.models.pyclone.binomial.ModelUpdater(feat_alloc_updater)


def simulate_data(D, N, K=None, alpha=1, eps=1e-3):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(K=K)

    Z = feat_alloc_dist.rvs(alpha, N)

    K = Z.shape[1]

    V = scipy.stats.gamma.rvs(1, 1, size=(K, D))

    params = pgfa.models.pyclone.binomial.Parameters(alpha, np.ones(2), V, np.ones(2), Z)

    F = params.F

    data = []

    cn_n = 2

    cn_r = 2

    mu_n = eps

    mu_r = eps

    t = np.ones(D)

    for n in range(N):
        phi = Z[n] @ F

        cn_total = 2  # scipy.stats.poisson.rvs(1) + 1

        cn_major = scipy.stats.randint.rvs(1, cn_total + 1)

        cn_minor = cn_total - cn_major

        cn_var = scipy.stats.randint.rvs(1, cn_major + 1)

        sample_data_points = []

        for d in range(D):
            mu_v = min(cn_var / cn_total, 1 - eps)

            xi = (1 - t[d]) * phi[d] * cn_n * mu_n + t[d] * (1 - phi[d]) * cn_r * mu_r + t[d] * phi[d] * cn_total * mu_v

            xi /= (1 - t[d]) * phi[d] * cn_n + t[d] * (1 - phi[d]) * cn_r + t[d] * phi[d] * cn_total

            d = scipy.stats.poisson.rvs(1000)

            b = scipy.stats.binom.rvs(d, xi)

            a = d - b

            sample_data_points.append(
                pgfa.models.pyclone.utils.get_sample_data_point(a, b, cn_major, cn_minor, 2, eps, 1.0)
            )

        data.append(
            pgfa.models.pyclone.utils.DataPoint(sample_data_points)
        )

    return data, params


if __name__ == '__main__':
    main()
