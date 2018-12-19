import numpy as np
from scanorama import plt
from scipy.sparse import csr_matrix
from time import time

from geosketch import *
from utils import log

if __name__ == '__main__':

    X_dimred = np.loadtxt('data/dimred/svd_zeisel.txt')
    X_dimred = X_dimred[:500000]
    
    sizes = [ 1000000, 2500000, 5000000, 10000000 ]
    times = []

    for size in sizes:
    
        X = np.repeat(X_dimred, size / X_dimred.shape[0], axis=1)

        t0 = time()
        gs(X, 20000)
        t1 = time()

        times.append(t1 - t0)

        print(t1 - t0)

    times = np.array(times) / 60.

    plt.figure()
    plt.plot(sizes, times)
    plt.scatter(sizes, times)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample size')
    plt.ylabel('Time (minutes)')
    plt.savefig('time_benchmark.svg')
