import numpy as np

from .base import AbstractTraceReader, AbstractTraceWriter


class TraceReader(AbstractTraceReader):
    def _get_param_trace(self, name):
        if name == 'gamma':
            trace = self._fh[name][:self.num_iters]

        elif name == 'F':
            trace = self._get_F()

        elif name == 'S':
            trace = self._get_S()

        elif name == 'V':
            trace = self._get_V()

        elif name == 'Z':
            trace = self._get_V()

        return trace

    def _get_F(self):
        N = self.data.shape[1]

        trace = []

        for idx in range(self.num_iters):
            K = self._fh['K'][idx]

            trace.append(
                self._fh['F'][idx].reshape((K, N))
            )

        return trace

    def _get_S(self):
        trace = []

        for idx in range(self.num_iters):
            trace.append(
                self._fh['S'][idx]
            )

        return trace

    def _get_V(self):
        D = self.data.shape[0]

        trace = []

        for idx in range(self.num_iters):
            K = self._fh['K'][idx]

            trace.append(
                self._fh['V'][idx].reshape((D, K))
            )

        return trace

    def _get_Z(self):
        D = self.data.shape[0]

        trace = []

        for idx in range(self.num_iters):
            K = self._fh['K'][idx]

            trace.append(
                self._fh['Z'][idx].reshape((D, K))
            )

        return trace


class TraceWriter(AbstractTraceWriter):
    def _init(self):
        self._init_dataset('gamma', np.float64)

        self._init_dataset('F', np.float64, vlen=True)

        self._init_dataset('S', np.float64)

        self._init_dataset('V', np.float64, vlen=True)

    def _write_row(self, model):
        self._fh['gamma'][self._iter] = model.params.gamma

        self._fh['F'][self._iter] = model.params.F.flatten()

        self._fh['S'][self._iter] = model.params.S

        self._fh['V'][self._iter] = model.params.V.flatten()
