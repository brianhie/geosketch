import numpy as np
from scanorama import plt
from scipy.sparse import csr_matrix
from time import time

from experiments import dropclust_preprocess, dropclust_sample
from save_mtx import save_mtx
from sketch import *
from utils import log

if __name__ == '__main__':

    sizes = [ 10000, 100000, 1000000, 10000000 ]
    times = []

    for size in sizes:
    
        X = np.random.normal(size=(size, 100))

        t0 = time()
        gs(X, 10000)
        t1 = time()

        times.append(t1 - t0)

    times = np.array(times) / 60.

    plt.figure()
    plt.plot(sizes, times)
    plt.scatter(sizes, times)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample size')
    plt.ylabel('Time (minutes)')
    plt.savefig('time_benchmark.svg')
