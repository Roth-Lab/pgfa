from .gibbs import GibbsUpdater


class GibbsMixtureUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.gibbs_updater = GibbsUpdater()

        self.other_updater = feat_alloc_updater

    def update(self, data, dist, feat_alloc_prior, params):
        params = self.gibbs_updater.update(data, dist, feat_alloc_prior, params)

        params = self.other_updater.update(data, dist, feat_alloc_prior, params)

        return params
