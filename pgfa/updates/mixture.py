from pgfa.math_utils import bernoulli_rvs
from pgfa.updates.gibbs import GibbsUpdater


class GibbsMixtureUpdater(object):

    def __init__(self, feat_alloc_updater, gibbs_prob=0.5):
        self.gibbs_updater = GibbsUpdater(
            annealing_schedule=feat_alloc_updater.annealing_schedule,
            singletons_updater=feat_alloc_updater.singletons_updater
        )

        self.other_updater = feat_alloc_updater

        self.gibbs_prob = gibbs_prob

    def update(self, model):
        if bernoulli_rvs(self.gibbs_prob) == 1:
            self.gibbs_updater.update(model)

        else:
            self.other_updater.update(model)
