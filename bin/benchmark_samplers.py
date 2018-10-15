import numpy as np
from scipy.sparse import csr_matrix

from dropclust_experiments import *
from save_mtx import save_mtx
from sketch import *
from utils import log

N = 10000

if __name__ == '__main__':
    X = np.random.normal(size=(1000000, 1000))

    log('Geometric sampling...')
    gs(X, N)
    log('Geometric sampling done.')

    log('Spatial random sampling...')
    srs(X, N)
    log('Spatial random sampling done.')

    name = 'data/benchmark_samplers'
    save_mtx(name, csr_matrix(X),
             [ str(i) for i in range(X.shape[1]) ])
    log('dropClust sampling...')
    dropclust_preprocess(name)
    dropclust_sample(name, N)
    log('dropClust sampling done.')
