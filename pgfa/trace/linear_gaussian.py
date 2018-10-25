import numpy as np

from pgfa.models.linear_gaussian import Parameters

from .base import AbstractTraceReader, AbstractTraceWriter


class TraceReader(AbstractTraceReader):
    def get_params_iter(self):
        for idx in range(self.num_iters):
            yield Parameters(
                tau_v, tau_x, V, Z)

    def _get_param_trace(self, name):
        if name in ['tau_v', 'tau_x', 'log_p_collapsed']:
            trace = self._fh[name][:self.num_iters]

        elif name == 'V':
            trace = self._get_V()

        elif name == 'Z':
            trace = self._get_Z()

        return trace

    def _get_V(self):
        D = self.data.shape[1]

        trace = []

        for idx in range(self.num_iters):
            K = self._fh['K'][idx]

            trace.append(
                self._fh['V'][idx].reshape((K, D))
            )

        return trace

    def _get_Z(self):
        N = self.data.shape[0]

        trace = []

        for idx in range(self.num_iters):
            K = self._fh['K'][idx]

            trace.append(
                self._fh['Z'][idx].reshape((N, K))
            )

        return trace


class TraceWriter(AbstractTraceWriter):
    def _init(self):
        self._init_dataset('tau_v', np.float64)

        self._init_dataset('tau_x', np.float64)

        self._init_dataset('V', np.float64, vlen=True)

    def _write_row(self, model):
        self._fh['tau_v'][self._iter] = model.params.tau_v

        self._fh['tau_x'][self._iter] = model.params.tau_x

        self._fh['V'][self._iter] = model.params.V.flatten()
