import numpy as np

from .base import TraceReader, TraceWriter


class LinearGaussianTraceReader(TraceReader):
    def _get_param_trace(self, name):
        if name in ['alpha', 'tau_v', 'tau_x', 'K', 'log_p_collapsed']:
            trace = self._fh[name][:self.num_iters]

        elif name == 'V':
            D = self.data.shape[1]

            trace = []

            for idx in range(self.num_iters):
                K = self._fh['K'][idx]

                trace.append(
                    self._fh[name][idx].reshape((K, D))
                )

        elif name == 'Z':
            N = self.data.shape[0]

            trace = []

            for idx in range(self.num_iters):
                K = self._fh['K'][idx]

                trace.append(
                    self._fh[name][idx].reshape((N, K))
                )

        return trace


class LinearGaussianTraceWriter(TraceWriter):
    def _init(self):
        self._init_dataset('log_p_collapsed', np.float64)

        self._init_dataset('K', np.int64)

        self._init_dataset('alpha', np.float64)

        self._init_dataset('tau_v', np.float64)

        self._init_dataset('tau_x', np.float64)

        self._init_dataset('V', np.float64, vlen=True)

        self._init_dataset('Z', np.int8, vlen=True)

    def _write_row(self, model):
        self._fh['log_p_collapsed'] = model.log_p_collapsed

        self._fh['K'][self._iter] = model.params.K

        if hasattr(model, 'alpha'):
            self._fh['alpha'][self._iter] = model.feat_alloc_prior.alpha

        self._fh['tau_v'][self._iter] = model.params.tau_v

        self._fh['tau_x'][self._iter] = model.params.tau_x

        self._fh['V'][self._iter] = model.params.V.flatten()

        self._fh['Z'][self._iter] = model.params.Z.flatten()
