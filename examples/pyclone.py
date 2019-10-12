import numpy as np

from pgfa.utils import get_b_cubed_score, get_feat_alloc_updater, set_seed, Timer

import pgfa.models.pyclone.binomial
import pgfa.models.pyclone.singletons_updates


def main(args):
    if args.ibp:
        print('Warning: IBP sampling for the PyClone model is not properly supported.')
    
    set_seed(args.data_seed)

    params = pgfa.models.pyclone.binomial.simulate_params(
        args.num_dims, args.num_data_points, K=args.num_features, alpha=args.alpha
    )

    data = pgfa.models.pyclone.binomial.simulate_data(params)

    model_updater = get_model_updater(
        annealing_power=args.annealing_power,
        feat_alloc_updater_type=args.sampler,
        ibp=args.ibp,
        mixture_prob=args.mixture_prob,
        num_particles=args.num_particles,
        test_path=args.test_path
    )

    set_seed(args.param_seed)

    if args.ibp:
        model_K = None

    else:
        model_K = args.num_features

    model = pgfa.models.pyclone.binomial.get_model(data, K=model_K)

    set_seed(args.run_seed)

    old_params = model.params.copy()

    model.params = params.copy()

    log_p_true = model.log_p

    model.params = old_params.copy()

    print('Arguments')

    print('-' * 100)

    for key, value in sorted(vars(args).items()):
        print('{0}: {1}'.format(key, value))

    print('@' * 100)

    print('True feature counts (sorted): {}'.format(sorted(np.sum(params.Z, axis=0))))

    print('True log density: {}'.format(log_p_true))

    print('@' * 100)

    timer = Timer()

    i = 0

    last_print_time = -np.float('inf')

    while timer.elapsed < args.time:
        if (timer.elapsed - last_print_time) > args.print_freq:
            last_print_time = timer.elapsed

            print('Iteration: {}'.format(i))

            print('Log density: {}'.format(model.log_p))

            print('Relative log density: {}'.format((model.log_p - log_p_true) / abs(log_p_true)))

            if args.ibp:
                print('Num features: {}'.format(model.params.K))

            print('B-Cube scores: {}'.format(get_b_cubed_score(params.Z, model.params.Z)))

            print('Feature counts (sorted): {}'.format(sorted(np.sum(model.params.Z, axis=0))))

            print('#' * 100)

        timer.start()

        model_updater.update(model)

        timer.stop()

        i += 1


def get_model_updater(
        feat_alloc_updater_type='g',
        annealing_power=0.0,
        ibp=True,
        mixture_prob=0.0,
        num_particles=20,
        test_path='zeros'):

    if ibp:
        singletons_updater = pgfa.models.pyclone.singletons_updates.PriorSingletonsUpdater()

    else:
        singletons_updater = None

    updater_kwargs = {'singletons_updater': singletons_updater}

    if feat_alloc_updater_type in ['dpf', 'pg']:
        updater_kwargs['annealing_power'] = annealing_power

        updater_kwargs['num_particles'] = num_particles

        updater_kwargs['test_path'] = test_path

    feat_alloc_updater = get_feat_alloc_updater(mixture_prob, feat_alloc_updater_type, updater_kwargs)

    return pgfa.models.pyclone.binomial.ModelUpdater(feat_alloc_updater)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    #===================================================================================================================
    # Dataset parameters
    #===================================================================================================================
    parser.add_argument(
        '-D', '--num-dims', default=10, type=int,
        help='''Number of dimensions of data.'''
    )

    parser.add_argument(
        '-K', '--num-features', default=5, type=int,
        help='''Number of features to use. If --ibp is not set this will be used for simulation and fitting.'''
    )

    parser.add_argument(
        '-N', '--num-data-points', default=100, type=int,
        help='''Number of data points to simulate.'''
    )

    #===================================================================================================================
    # Model parameters
    #===================================================================================================================
    parser.add_argument(
        '--alpha', default=1.0, type=float,
        help='''alpha parameter for the feature allocation prior.'''
    )

    parser.add_argument(
        '--ibp', action='store_true', default=False,
        help='''If set then IBP model will be fit.'''
    )

    #===================================================================================================================
    # Sampler options
    #===================================================================================================================
    parser.add_argument(
        '-s', '--sampler', choices=['dpf', 'g', 'pg', 'rg'], default='g',
        help='''Sampler used to fit the feature allocation model.
        Choices are: `dpf`-Discrete Particle Filter, `g`-Gibbs, `pg`-Particle Gibbs, `rg`-Row Gibbs
        '''
    )

    parser.add_argument(
        '-t', '--time', default=100, type=float,
        help='''How long the model fitting will run.'''
    )

    parser.add_argument(
        '--mixture-prob', default=0.0, type=float,
        help='''Probability of performing a Gibbs update when mixing particle and Gibbs samplers.'''
    )

    parser.add_argument(
        '--print-freq', default=10, type=float,
        help='''How often sampler status should be printed.'''
    )

    #===================================================================================================================
    # Particle sampler options
    #===================================================================================================================
    parser.add_argument(
        '--annealing-power', default=1.0, type=float,
        help='''Annealing power for target densities. Has no effect for Gibbs sampler.'''
    )

    parser.add_argument(
        '--num-particles', default=20, type=int,
        help='''Number of particles to use for dpf or pg samplers. Has no effect for Gibbs sampler.'''
    )

    parser.add_argument(
        '--test-path',
        choices=['conditional', 'ones', 'random', 'two-stage', 'unconditional', 'zeros'],
        default='zeros',
        help='''Strategy for setting test path for particle sampler. Has no effect for Gibbs sampler.'''
    )

    #===================================================================================================================
    # Random seeds
    #===================================================================================================================
    parser.add_argument(
        '--data-seed', default=None, type=int,
        help='''Random seed for simulating data.'''
    )

    parser.add_argument(
        '--param-seed', default=None, type=int,
        help='''Random seed for simulating initial parameters.'''
    )

    parser.add_argument(
        '--run-seed', default=None, type=int,
        help='''Random seed for running sampler.'''
    )

    args = parser.parse_args()

    main(args)
