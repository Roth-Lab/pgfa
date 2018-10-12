from .gibbs import GibbsUpdater


class GibbsMixtureUpdater(object):
    def __init__(self, feat_alloc_updater):
        self.gibbs_updater = GibbsUpdater()

        self.other_updater = feat_alloc_updater

    def update(self, model):
        self.gibbs_updater.update(model)

        self.other_updater.update(model)
