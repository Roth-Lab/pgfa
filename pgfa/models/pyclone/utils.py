import numba
import numpy as np

from pgfa.math_utils import log_normalize


def get_sample_data_point(a, b, cn_major, cn_minor, cn_normal=2, error_rate=1e-3, tumour_content=1.0):
    cn_total = cn_major + cn_minor

    cn = []

    mu = []

    log_pi = []

    # Consider all possible mutational genotypes consistent with mutation before CN change
    for x in range(1, cn_major + 1):
        cn.append((cn_normal, cn_normal, cn_total))

        mu.append((error_rate, error_rate, min(1 - error_rate, x / cn_total)))

        log_pi.append(0)

    # Consider mutational genotype of mutation before CN change if not already added
    mutation_after_cn = (cn_normal, cn_total, cn_total)

    if mutation_after_cn not in cn:
        cn.append(mutation_after_cn)

        mu.append((error_rate, error_rate, min(1 - error_rate, 1 / cn_total)))

        log_pi.append(0)

    cn = np.array(cn, dtype=np.int)

    mu = np.array(mu, dtype=np.float)

    log_pi = log_normalize(np.array(log_pi, dtype=np.float64))

    return SampleDataPoint(int(a), int(b), cn, mu, log_pi, tumour_content)


class DataPoint(object):

    def __init__(self, sample_data_points):
        self.sample_data_points = sample_data_points


@numba.experimental.jitclass([
    ('b', numba.int64),
    ('d', numba.int64),
    ('cn', numba.int64[:, :]),
    ('mu', numba.float64[:, :]),
    ('log_pi', numba.float64[:]),
    ('tumour_content', numba.float64)
])
class SampleDataPoint(object):

    def __init__(self, a, b, cn, mu, log_pi, tumour_content=1.0):
        self.b = b
        self.d = a + b
        self.cn = cn
        self.mu = mu
        self.log_pi = log_pi
        self.tumour_content = tumour_content
