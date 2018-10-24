import h5py
import numpy as np


class AbstractTraceReader(object):
    def _get_param_trace(self, name):
        raise NotImplementedError()

    def __init__(self, file_name):
        self._fh = h5py.File(file_name, mode='r')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, name):
        if name in ['alpha', 'log_p', 'time', 'K']:
            trace = self._fh[name][:self.num_iters]

        elif self._is_valid_param(name):
            trace = self._get_param_trace(name)

        else:
            raise KeyError('Invalid trace parameter {}'.format(name))

        return trace

    @property
    def data(self):
        return self._fh['data'].value

    @property
    def num_iters(self):
        return self._fh['iter'][0]

    def close(self):
        self._fh.close()

    def _is_valid_param(self, name):
        return (name in list(self._fh.keys())) and (name not in ['data', 'iter'])


class AbstractTraceWriter(object):
    def _write_row(self, model):
        raise NotImplementedError()

    def _init(self):
        raise NotImplementedError()

    def __init__(self, file_name, model):
        self._fh = h5py.File(file_name, 'w')

        self._iter = 0

        self._max_size = 1000

        self._fh.create_dataset('data', data=model.data)

        self._fh.create_dataset('iter', (1,), dtype=np.int64)

        self._init_dataset('log_p', np.float64)

        self._init_dataset('time', np.float64)

        self._init_dataset('K', np.int64)

        self._init_dataset('Z', np.int64, vlen=True)

        if hasattr(model.feat_alloc_prior, 'alpha'):
            self._init_dataset('alpha', np.float64)

        self._init()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._fh.close()

    def write_row(self, model, time):
        self._resize_if_needed()

        self._fh['iter'][0] = self._iter

        self._fh['log_p'][self._iter] = model.log_p

        self._fh['time'][self._iter] = time

        self._fh['K'][self._iter] = model.params.Z.shape[1]

        self._fh['Z'][self._iter] = model.params.Z.flatten()

        if hasattr(model.feat_alloc_prior, 'alpha'):
            self._fh['alpha'][self._iter] = model.feat_alloc_prior.alpha

        self._write_row(model)

        self._iter += 1

    def _init_dataset(self, name, dtype, vlen=False):
        if vlen:
            dtype = h5py.special_dtype(vlen=dtype)

        self._fh.create_dataset(
            name,
            (self._max_size,),
            compression='gzip',
            dtype=dtype,
            maxshape=(None,)
        )

    def _resize_if_needed(self):
        if self._iter >= self._max_size:
            self._max_size *= 2

            for name in self._fh.keys():
                if name in ['data', 'iter']:
                    continue

                self._fh[name].resize((self._max_size,))
