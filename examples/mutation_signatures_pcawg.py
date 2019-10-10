from scipy.special import logsumexp as log_sum_exp

import numba
import numpy as np
import pandas as pd

import pgfa.feature_allocation_distributions
import pgfa.models.mutation_signatures
import pgfa.updates

from pgfa.utils import Timer

import warnings
warnings.filterwarnings("ignore")


def main():
    print_freq = 10

    data_seed = 1
    param_seed = 0
    run_seed = 1

    updater = 'udpf'
    test_path = 'random'

    time = np.inf

    set_seed(data_seed)

    data_file = '/home/andrew/projects/pgfa/data/mutation_signatures/data.tsv'
    sigs_file = '/home/andrew/projects/pgfa/data/mutation_signatures/signatures.tsv'

    data_df = pd.read_csv(data_file, index_col=0, sep='\t')

    data = data_df.iloc[:100].values

    data_train, data_test = split_data(data)

    data_test = np.array(data_test, dtype=np.int)

    sigs_df = pd.read_csv(sigs_file, index_col=0, sep='\t')

    sigs_df = sigs_df[data_df.columns]

    model_updater = get_model_updater(
        annealing_power=1.0, feat_alloc_updater_type=updater, mixture_prob=0.0, num_particles=20, test_path=test_path
    )

    set_seed(param_seed)

    K = sigs_df.shape[0]

    model = get_model(data_train, K=K)

    model.params.S = sigs_df.values

    model.params.S = model.params.S / np.sum(model.params.S, axis=1)[:, np.newaxis]

#     model.params.V = np.ones(model.params.V.shape)

#     model.params.Z = np.ones(model.params.Z.shape, dtype=np.int8)

    print('@' * 100)

    set_seed(run_seed)

    timer = Timer()

    i = 0

    last_print_time = -np.float('inf')

    while timer.elapsed < time:
        if (timer.elapsed - last_print_time) > print_freq:
            last_print_time = timer.elapsed

            print(
                i,
                model.params.K,
                model.params.alpha,
                model.log_p
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params),
                compute_log_p(data_test, model),
                compute_log_predictive(data_test, model)
            )

            print(np.sum(model.params.Z, axis=0))

            print(model.params.W[0])

            print('#' * 100)

        timer.start()

        model_updater.update(model)

        timer.stop()

        i += 1


def get_model(data, K):
    feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.mutation_signatures.Model(data, feat_alloc_dist)


def get_model_updater(
        feat_alloc_updater_type='g', annealing_power=0, mixture_prob=0.0, num_particles=20, test_path='conditional'
    ):

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.ConditionalDiscreteParticleFilterRowUpdater(
            annealing_power=annealing_power, max_particles=num_particles, test_path=test_path
        )

    elif feat_alloc_updater_type == 'udpf':
        feat_alloc_updater = pgfa.updates.DiscreteParticleFilterRowUpdater(
            annealing_power=annealing_power, max_particles=num_particles, test_path='conditional'
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater()

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealing_power=annealing_power, num_particles=num_particles
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater()

    if mixture_prob > 0:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater, gibbs_prob=mixture_prob)

    return pgfa.models.mutation_signatures.ModelUpdater(feat_alloc_updater)


def set_seed(seed):
    np.random.seed(seed)

    set_numba_seed(seed)


@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)


def split_data(X):
    X = X.copy()

    N, _ = X.shape

    X_train = np.zeros(X.shape, dtype=np.int)

    for n in range(N):
        tot = np.sum(X[n])

        train_num = int(0.8 * tot)

        while train_num > 0:
            idxs = np.where(X[n] > 0)[0]

            d = np.random.choice(idxs)

            X_train[n, d] += 1

            X[n, d] -= 1

            train_num -= 1

    return X_train, X


def compute_log_p(data, model):
    return model.data_dist.log_p(data, model.params)


def compute_log_predictive(data, model):
    # K x D
    S = model.params.S
    # N x D
    X = data
    # N x K
    W = model.params.W
    # N x K x D
    log_p = np.log(W[:, :, np.newaxis] * S[np.newaxis, :, :] + 1e-10)
    # N x D
    log_p = log_sum_exp(log_p, axis=1)
    return np.sum(X * log_p) / np.sum(X)


if __name__ == '__main__':
    main()
