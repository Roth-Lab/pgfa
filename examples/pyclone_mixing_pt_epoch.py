import numba
import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial.distance

from pgfa.utils import get_b_cubed_score

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone
import pgfa.updates

from pgfa.utils import Timer
from pgfa.math_utils import do_metropolis_hastings_accept_reject


def main():
    seed = 1

    set_seed(seed)

    ibp = True
    time = 100000
    K = 4
    num_epochs = 8
    
    data, Z_true, mutations = load_data()
    
#     idxs = np.random.choice(len(data), 20, replace=False)
#     
#     data = [data[i] for i in idxs]
#     
#     Z_true = Z_true[idxs]

    model_updater = get_model_updater(
        feat_alloc_updater_type='dpf', ibp=ibp, mixed_updates=False
    )

    model = get_model(data, ibp=ibp, K=K)
    
    epoch = 0
    
    max_iters = 10
    
    for epoch in range(num_epochs):
        engine = ParallelTemperingEngine(model, model_updater, ladder_power=3.0, num_chains=2 ** epoch)

        print('@' * 100)
    
        timer = Timer()
    
        iter_idx = 0
    
        while timer.elapsed < time:            
            if iter_idx % 1 == 0:          
                print(
                    epoch,
                    iter_idx,
                    model.params.K,
                    model.params.alpha,
                    model.log_p,
                )
    
                print(
                    get_b_cubed_score(Z_true, model.params.Z)
                )
    
                print(
                    model.data_dist.log_p(model.data, model.params),
                    model.feat_alloc_dist.log_p(model.params),
                    model.params_dist.log_p(model.params)
                )
                
                print(model.params.Z.sum(axis=0))
                
                print(np.around(model.params.F, 3))
    
                print('#' * 100)
    
            timer.start()
            
            engine.update()
            
            timer.stop()
    
            iter_idx += 1
            
            if iter_idx > max_iters:
                break
            
        max_iters *= 2
        

class ParallelTemperingEngine(object):

    def __init__(self, model, model_updater, ladder_power=1.0, num_chains=10):
        self.model = model
        
        self.model_updater = model_updater
        
        self.num_chains = num_chains
        
        if self.num_chains == 1:
            self.annealing_ladder = np.array([1.0])
        
        else:
            self.annealing_ladder = np.linspace(0, 1, num_chains) ** ladder_power
        
#         self.params = [model.get_default_params(model.data, model.feat_alloc_dist) for _ in range(num_chains - 1)]
        
#         self.params.append(model.params.copy())
        
        self.params = [model.params.copy() for _ in range(num_chains)]
        
        self.iter_idx = 0
        
        self.sm_updater = pgfa.models.pyclone.SplitMergeUpdater()
        
    def update(self):
        for idx in range(self.num_chains):
            self.model.data_dist.annealing_power = self.annealing_ladder[idx]
            
            self.model.params = self.params[idx]  # .copy()

            self.model_updater.update(self.model)
            
            for _ in range(20):
                self.sm_updater.update(self.model)
            
            self.params[idx] = self.model.params  # .copy()
        
        self._do_chain_swap()
        
        self.model.data_dist.annealing_power = self.annealing_ladder[-1]
        
        self.model.params = self.params[-1]
        
        self.iter_idx += 1
            
    def _do_chain_swap(self):
        if self.iter_idx % 2 == 0:
            pairs = list(zip(np.arange(0, self.num_chains, 2), np.arange(1, self.num_chains, 2)))
        
        else:
            pairs = list(zip(np.arange(1, self.num_chains, 2), np.arange(2, self.num_chains, 2)))
        
        for i, j in pairs:            
            log_p_new = self._log_p(i, j) + self._log_p(j, i)
            
            log_p_old = self._log_p(i, i) + self._log_p(j, j)

            if do_metropolis_hastings_accept_reject(log_p_new, log_p_old, 0, 0):
                print('Swapping chains: {0} and {1}'.format(i, j))
                
                self.params[i], self.params[j] = self.params[j], self.params[i]
        
    def _log_p(self, chain_idx, param_idx):
        self.model.data_dist.annealing_power = self.annealing_ladder[chain_idx]
        
        return self.model.joint_dist.log_p(self.model.data, self.params[param_idx])


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

    return pgfa.models.pyclone.Model(data, feat_alloc_dist)


def get_model_updater(feat_alloc_updater_type='g', annealing_power=0, ibp=True, mixed_updates=False, num_particles=20):
    if ibp:
        singletons_updater = pgfa.models.pyclone.PriorSingletonsUpdater()

    else:
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

    return pgfa.models.pyclone.ModelUpdater(feat_alloc_updater)


def load_data():
    file_name = 'data/pyclone/mixing.tsv'
    
    df = pd.read_csv(file_name, sep='\t')
    
    Z = []
    
    cases = set()
    
    for x in df['variant_cases'].unique():
        cases.update(set(x.split(',')))
    
    cases = sorted(cases)
    
    samples = sorted(df['sample_id'].unique())
    
    data = []
    
    mutations = []
    
    for x, mut_df in df.groupby('mutation_id'):
        mutations.append(x)
        
        mut_df = mut_df.set_index('sample_id')
        
        sample_data_points = []
        
        for sample_id in samples:
            row = mut_df.loc[sample_id]
            
            sample_data_points.append(
                pgfa.models.pyclone.get_sample_data_point(row['a'], row['b'], row['cn_major'], row['cn_minor'])
            )
        
        data.append(
            pgfa.models.pyclone.DataPoint(sample_data_points)
        )
        
        Z_mut = np.zeros(4)
        
        for case in row['variant_cases'].split(','):
            Z_mut[cases.index(case)] = 1
        
        Z.append(Z_mut)
    
    Z = np.array(Z, dtype=np.int8)
    
    return data, Z, mutations


if __name__ == '__main__':
    main()
