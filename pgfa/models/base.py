import pgfa.feature_allocation_distributions


class AbstractModel(object):

    @staticmethod
    def get_default_params(data, feat_alloc_dist):
        raise NotImplementedError

    def _init_joint_dist(self, feat_alloc_dist):
        raise NotImplementedError

    def __init__(self, data, feat_alloc_dist, params=None):
        self.data = data

        if params is None:
            params = self.get_default_params(data, feat_alloc_dist)

        self.params = params

        self._init_joint_dist(feat_alloc_dist)

    @property
    def data_dist(self):
        return self.joint_dist.data_dist

    @property
    def feat_alloc_dist(self):
        return self.joint_dist.feat_alloc_dist

    @property
    def params_dist(self):
        return self.joint_dist.params_dist

    @property
    def log_p(self):
        """ Log of joint pdf.
        """
        return self.joint_dist.log_p(self.data, self.params)


class AbstractModelUpdater(object):

    def _update_model_params(self, model):
        """ Update the model specific parameters.
        """
        raise NotImplementedError

    def __init__(self, feat_alloc_updater):
        self.feat_alloc_updater = feat_alloc_updater

    def update(self, model, alpha_updates=1, feat_alloc_updates=1, param_updates=1):
        """ Update all parameters in a feature allocation model.
        """
        for _ in range(feat_alloc_updates):
            self.feat_alloc_updater.update(model)

        for _ in range(param_updates):
            self._update_model_params(model)

        for _ in range(alpha_updates):
            pgfa.feature_allocation_distributions.update_alpha(model)


class AbstractDataDistribution(object):

    def log_p(self, data, params):
        raise NotImplementedError

    def log_p_row(self, data, params, row_idx):
        raise NotImplementedError


class AbstractParametersDistribution(object):

    def log_p(self, params):
        raise NotImplementedError


class AbstractParameters(object):

    @property
    def param_shapes(self):
        raise NotImplementedError

    @property
    def D(self):
        """ Number of dimensions of dataset.
        """
        raise NotImplementedError

    @property
    def N(self):
        """ Number of data points.
        """
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    @property
    def K(self):
        """ Number of features.
        """
        return self.Z.shape[1]


class JointDistribution(object):

    def __init__(self, data_dist, feat_alloc_dist, params_dist):
        self.data_dist = data_dist

        self.feat_alloc_dist = feat_alloc_dist

        self.params_dist = params_dist

    def log_p(self, data, params):
        log_p = 0

        log_p += self.data_dist.log_p(data, params)

        log_p += self.feat_alloc_dist.log_p(params)

        log_p += self.params_dist.log_p(params)

        return log_p


class MAPJointDistribution(object):

    def __init__(self, data_dist, feat_alloc_dist, params_dist, temp=1e-3):
        self.data_dist = data_dist

        self.feat_alloc_dist = feat_alloc_dist

        self.params_dist = params_dist
        
        self.temp = temp

    def log_p(self, data, params):        
        log_p = 0

        log_p += self.data_dist.log_p(data, params)

        log_p += self.feat_alloc_dist.log_p(params)

        log_p += self.params_dist.log_p(params)

        return (1 / self.temp) * log_p
    
