import lda
import numba
import numpy as np
import pandas as pd

import pgfa.feature_allocation_distributions
import pgfa.models.mutation_signatures
import pgfa.updates

from pgfa.utils import Timer


def main():  
    annealing_power = 1.0
    num_particles = 20
    
    data_seed = 0
    param_seed = 1
    run_seed = 0
    updater = 'dpf'

    ibp = False
    time = np.inf

    set_seed(data_seed)

    data_file = '/home/andrew/projects/pgfa/data/mutation_signatures/data.tsv'
    sigs_file = '/home/andrew/projects/pgfa/data/mutation_signatures/signatures.tsv'
    
    data_df = pd.read_csv(data_file, index_col=0, sep='\t')
    
    data_true = data_df.iloc[:100].values
    
    data = data_true.copy().astype(float)
    
    data[np.random.random(data.shape) <= 0.1] = np.nan
    
#     data, data_test = split_data(data)
    
    sigs_df = pd.read_csv(sigs_file, index_col=0, sep='\t')
    
    sigs_df = sigs_df[data_df.columns]
    
    model_updater = get_model_updater(
        annealing_power=annealing_power, feat_alloc_updater_type=updater, ibp=ibp, mixed_updates=ibp, num_particles=num_particles
    )

    set_seed(param_seed)

    K = sigs_df.shape[0]

    model = get_model(data, ibp=ibp, K=K)
    
    model.params.S = sigs_df.values
    
    model.params.V = get_lda_weights(data_true, sigs_df.values) * K

    model.params.Z = np.ones(model.params.Z.shape, dtype=np.int8)
    
    print(data_df.shape)
    
    print('@' * 100)
    
    set_seed(run_seed)

    timer = Timer()

    i = 0

    while timer.elapsed < time:
        if i % 1 == 0:
            print(
                i,
                model.params.K,
                model.log_p
            )

            print(
                model.data_dist.log_p(model.data, model.params),
                model.feat_alloc_dist.log_p(model.params),
                model.params_dist.log_p(model.params),
#                 model.data_dist.log_p(data_test, model.params),
                compute_error(data_true, model),
#                 compute_error(data_test, model)
            )
            
            print(np.sum(model.params.Z, axis=0))
            
            print(model.params.W[0])

            print('#' * 100)

        timer.start()

        model_updater.update(model, param_updates=0)

        timer.stop()

        i += 1


def get_lda_weights(data, sigs):
    model = lda.LDA(n_topics=30, n_iter=1500, random_state=0)
    
    model.components_ = sigs

    return model.transform(data)


def get_model(data, ibp=False, K=None):
    if ibp:
        feat_alloc_dist = pgfa.feature_allocation_distributions.IndianBuffetProcessDistribution()
    
    else:
        feat_alloc_dist = pgfa.feature_allocation_distributions.BetaBernoulliFeatureAllocationDistribution(K)

    return pgfa.models.mutation_signatures.Model(data, feat_alloc_dist)

# def get_min_reconstruction_error(pred_sigs, true_sigs):
#     K = pred_sigs.shape[0]
#     
#     for k in range(K):


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
    singletons_updater = None

    if feat_alloc_updater_type == 'dpf':
        feat_alloc_updater = pgfa.updates.DicreteParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'g':
        feat_alloc_updater = pgfa.updates.GibbsUpdater(singletons_updater=singletons_updater)
        
    elif feat_alloc_updater_type == 'gpf':
        feat_alloc_updater = pgfa.updates.GumbelParticleFilterUpdater(
            annealing_power=annealing_power, max_particles=num_particles
            )

    elif feat_alloc_updater_type == 'pg':
        feat_alloc_updater = pgfa.updates.ParticleGibbsUpdater(
            annealing_power=annealing_power, num_particles=num_particles, singletons_updater=singletons_updater
        )

    elif feat_alloc_updater_type == 'rg':
        feat_alloc_updater = pgfa.updates.RowGibbsUpdater(singletons_updater=singletons_updater)

    if mixed_updates:
        feat_alloc_updater = pgfa.updates.GibbsMixtureUpdater(feat_alloc_updater)

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
        
        train_num = int(0.5 * tot) 
        
        while train_num > 0:
            idxs = np.where(X[n] > 0)[0]
            
            d = np.random.choice(idxs)
            
            X_train[n, d] += 1
            
            X[n, d] -= 1
            
            train_num -= 1
        
    return X_train, X


def compute_error(data, model):
    pi_hat = data / np.sum(data, axis=1)[:, np.newaxis]
    
    return np.sum(np.abs(model.params.pi - pi_hat))
            

if __name__ == '__main__':
    main()
