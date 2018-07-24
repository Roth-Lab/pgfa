import numpy as np


class LinearSumDistribution(object):
    def __init__(self, model):
        self.model = model

    def log_p(self, data, feat_mat, feat_params):
        log_p = 0

        for k in range(self.model.num_features):
            log_p += self.model.feature_prior_dist.log_p(feat_params[k], self.model.feature_prior_params[k])

        # Likelihood
        for n in range(self.model.num_data_points):
            x = data[n]

            z = feat_mat[n]

            f = np.sum(feat_params * z[:, np.newaxis], axis=0)

            log_p += self.model.feature_dist.log_p(x, f)

        return log_p


class FiniteFeatureAllocationModel(object):
    def __init__(
            self,
            data,
            feature_dist,
            feature_prior_dist,
            feature_prior_params,
            feature_params=None,
            feature_weight_params=None,
            latent_values=None,
            num_features=1):

        self.data = data

        self.feature_dist = feature_dist

        self.feature_prior_dist = feature_prior_dist

        self.num_features = num_features

        self._init_feature_prior_params(feature_prior_params)

        self._init_feature_params(feature_params)

        self._init_feature_weights_params(feature_weight_params)

        self._init_latent_values(latent_values)

    @property
    def feature_params_dim(self):
        return self.feature_params.shape[1]

    @property
    def log_p(self):
        log_p = 0

        N = self.num_data_points

        m = np.sum(self.latent_values, axis=0)

        a = self.feature_weight_params[:, 0]

        b = self.feature_weight_params[:, 1]

        a_new = a + m

        b_new = b + (N - m)

        for k in range(self.num_features):
            log_p += log_beta(a_new[k], b_new[k]) - log_beta(a[k], b[k])

            log_p += self.feature_prior_dist.log_p(self.feature_params[k], self.feature_prior_params[k])

        # Likelihood
        for n in range(self.num_data_points):
            x = self.data[n]

            z = self.latent_values[n]

            f = np.sum(self.feature_params * z[:, np.newaxis], axis=0)

            log_p += self.feature_dist.log_p(x, f)

        return log_p

    @property
    def num_data_points(self):
        return self.data.shape[0]

    def _init_feature_params(self, params):
        if params is None:
            params = []

            for k in range(self.num_features):
                params.append(self.feature_prior_dist.rvs(self.feature_prior_params))

            params = np.squeeze(params)

            if params.ndim == 1:
                params = params.reshape((self.num_features, 1))

        self.feature_params = params

    def _init_feature_prior_params(self, params):
        params = np.atleast_2d(params)

        self.feature_prior_params = params

    def _init_feature_weights_params(self, params):
        if params is None:
            params = np.ones(2)

        self.feature_weight_params = params

    def _init_latent_values(self, params):
        if params is None:
            params = np.random.randint(0, 2, (self.num_data_points, self.num_features))

        self.latent_values = params
