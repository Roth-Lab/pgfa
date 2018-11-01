import h5py
import json
import numpy as np


class TraceReader(object):
    def __init__(self, file_name):
        self._fh = h5py.File(file_name, 'r')

        self._trace_shape_attrs = {}

        for name in self._fh.keys():
            if 'shape' not in self._fh[name].attrs:
                self._trace_shape_attrs[name] = ()

            else:
                self._trace_shape_attrs[name] = list(json.loads(self._fh[name].attrs['shape']))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        trace = {}

        trace_shape = {}

        for name in self._fh.keys():
            trace[name] = self._fh[name].value

            trace_shape[name] = self._get_trace_shape

        for idx in range(self.num_iters):
            K = trace['K'][idx]

            row = {}

            row['K'] = K

            for name in self._fh.keys():
                if name in ['data', 'iter', 'D', 'K', 'N']:
                    continue

                elif name == 'time':
                    row[name] = trace[name][idx]

                else:
                    shape = self._get_trace_shape(self._trace_shape_attrs[name], K)

                    row[name] = trace[name][idx].reshape(shape)

            yield row

    @property
    def data(self):
        return self._fh['data'].value

    @property
    def num_iters(self):
        return self._fh['iter'][()]

    @property
    def D(self):
        return self._fh['D'][()]

    @property
    def N(self):
        return self._fh['N'][()]

    def close(self):
        self._fh.close()

    def get_iter_trace(self, idx):
        row = {}

        K = self._fh['K'][idx]

        row['K'] = K

        for name in self._fh.keys():
            if name in ['data', 'iter', 'D', 'K', 'N']:
                continue

            elif name == 'time':
                row[name] = self._fh[name][idx]

            else:
                shape = self._get_trace_shape(self._trace_shape_attrs[name], K)

                row[name] = self._fh[name][idx].reshape(shape)

        return row

    def _get_trace_shape(self, shape_attr, K):
        shape_map = {'D': self.D, 'K': K, 'N': self.N}

        shape = list(shape_attr)

        for i, x in enumerate(shape):
            if x in shape_map:
                shape[i] = shape_map[x]

        return shape


class TraceWriter(object):
    def __init__(self, file_name, model):
        self._fh = h5py.File(file_name, 'w')

        self._iter = 0

        self._max_size = 10

        self._fh.create_dataset('data', compression='gzip', data=model.data)

        self._fh.create_dataset('iter', data=0, dtype=np.int64)

        self._fh.create_dataset('D', data=model.params.D)

        self._fh.create_dataset('N', data=model.params.N)

        self._fh.create_dataset('K', (self._max_size,), dtype=np.int64, maxshape=(None,))

        self._fh.create_dataset('time', (self._max_size,), dtype=np.float64, maxshape=(None,))

        for name, shape in model.params.param_shapes.items():
            p = getattr(model.params, name)

            if isinstance(p, np.ndarray):
                dtype = p.dtype

            else:
                dtype = type(p)

            if 'K' in shape:
                dtype = h5py.special_dtype(vlen=dtype)

            trace_shape = self._get_trace_shape(shape)

            max_trace_shape = list(trace_shape)

            max_trace_shape[0] = None

            self._fh.create_dataset(
                name, trace_shape, compression='gzip', dtype=dtype, maxshape=max_trace_shape
            )

            self._fh[name].attrs['shape'] = json.dumps(shape)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def D(self):
        return self._fh['D'][0]

    @property
    def N(self):
        return self._fh['N'][0]

    def close(self):
        self._fh.close()

    def write_row(self, model, time):
        self._resize_if_needed()

        self._fh['iter'][()] = self._iter

        self._fh['time'][self._iter] = time

        self._fh['K'][self._iter] = model.params.K

        for name, shape in model.params.param_shapes.items():
            p = getattr(model.params, name)

            if 'K' in shape:
                p = p.flatten()

            self._fh[name][self._iter] = p

        self._iter += 1

    def _get_trace_shape(self, shape):
        if len(shape) == 0:
            trace_shape = (self._max_size,)

        elif 'K' in shape:
            trace_shape = (self._max_size,)

        else:
            trace_shape = [self._max_size]

            for x in shape:
                if x == 'D':
                    trace_shape.append(self.D)

                elif x == 'N':
                    trace_shape.append(self.N)

                else:
                    trace_shape.append(x)

        return trace_shape

    def _resize_if_needed(self):
        if self._iter >= self._max_size:
            self._max_size *= 2

            for name in self._fh.keys():
                if name in ['data', 'iter', 'D', 'N']:
                    continue

                shape = list(self._fh[name].shape)

                shape[0] = self._max_size

                self._fh[name].resize(shape)
