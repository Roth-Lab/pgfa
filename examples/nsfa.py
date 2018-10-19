import numpy as np

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_priors
import pgfa.models.nsfa
import pgfa.updates


def main():
    np.random.seed(0)

    num_iters = 100001
    D = 10
    K = 5
    N_train = 1000
    N_test = 1000

    feat_alloc_prior = pgfa.feature_allocation_priors.BetaBernoulliFeatureAllocationDistribution(1, 1, K)

    feat_alloc_prior = pgfa.feature_allocation_priors.IndianBuffetProcessDistribution()

    params_train = get_test_params(feat_alloc_prior, N_train, D, seed=0)

    params_test = get_test_params(feat_alloc_prior, N_test, D, params=params_train, seed=1)

    data_train = get_data(params_train)

    data_test = get_data(params_test)

    singletons_updater = pgfa.models.nsfa.PriorSingletonsUpdater()

    singletons_updater = pgfa.models.nsfa.CollapsedSingletonsUpdater()

    feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(
        pgfa.updates.ParticleGibbsUpdater(
            annealed=True,  num_particles=10, singletons_updater=singletons_updater
        )
    )

#     feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)

    model_updater = pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModelUpdater(feat_alloc_updater)

    feat_alloc_prior = pgfa.feature_allocation_priors.IndianBuffetProcessDistribution()

    model = pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModel(data_train, feat_alloc_prior)

    model.params.F = np.random.normal(0, 1, size=(1, N_train))

    model.params.V = np.random.normal(0, 1, size=(D, 1))

    model.params.Z = np.random.randint(0, 2, size=(D, 1))

#     model.params = params_train.copy()

    print(np.sum(params_train.Z, axis=0))

    print('@' * 100)

    for i in range(num_iters):
        if i % 100 == 0:
            print(
                i,
                model.params.Z.shape[1],
                model.params.gamma,
                model.log_p,
                model.log_predictive_pdf(data_test),
                model.rmse,
            )

            if model.params.K > 0:
                try:
                    print(get_b_cubed_score(params_train.Z, model.params.Z))
                except:
                    pass

            print(np.sum(model.params.Z, axis=0))

            print('#' * 100)

        model_updater.update(model)


def get_data(params):
    data = params.W @ params.F

    data += np.random.multivariate_normal(np.zeros(params.D), np.diag(1 / params.S), size=params.N).T

    return data


def get_test_params(feat_alloc_prior, num_data_points, num_observed_dims, params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    D = num_observed_dims
    N = num_data_points

    if params is None:
        gamma = 1

        Z = feat_alloc_prior.rvs(D)

        K = Z.shape[1]

        S = 1 * np.ones(D)

        V = np.random.multivariate_normal(np.zeros(K), (1 / gamma) * np.eye(K), size=D)

        F = np.random.normal(0, 1, size=(K, N))

    else:
        gamma = params.gamma

        Z = params.Z.copy()

        K = Z.shape[1]

        S = params.S.copy()

        V = params.V.copy()

        F = np.random.normal(0, 1, size=(K, N))

    return pgfa.models.nsfa.Parameters(gamma, F, S, V, Z)


def get_min_error(params_pred, params_true):
    import itertools

    W_pred = params_pred.F.T

    W_true = params_true.F.T

    K_pred = W_pred.shape[1]

    K_true = W_true.shape[1]

    min_error = float('inf')

    for perm in itertools.permutations(range(K_pred)):
        error = np.sqrt(np.mean((W_pred[:, perm[:K_true]] - W_true)**2))

        if error < min_error:
            min_error = error

    return min_error


if __name__ == '__main__':
    #     import line_profiler
    #     import pgfa.updates.particle_gibbs
    #
    #     profiler = line_profiler.LineProfiler(
    #         pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModelUpdater.update,
    #         pgfa.updates.particle_gibbs.do_particle_gibbs_update,
    #         pgfa.updates.particle_gibbs._propose_annealed,
    #         pgfa.updates.particle_gibbs._log_target_pdf_annealed
    #     )
    #
    #     profiler.run("main()")
    #
    #     profiler.print_stats()

    main()
