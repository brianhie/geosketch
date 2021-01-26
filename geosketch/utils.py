import errno
from fbpca import pca
import datetime
import numpy as np
import os
from sklearn.random_projection import SparseRandomProjection as JLSparse
import sys

# Default parameters.
DIMRED = 100

def log(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | [geosketch] ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def reduce_dimensionality(X, method='svd', dimred=DIMRED, raw=False):
    if method == 'svd':
        k = min((dimred, X.shape[0], X.shape[1]))
        U, s, Vt = pca(X, k=k, raw=raw)
        return U[:, range(k)] * s[range(k)]
    elif method == 'jl_sparse':
        jls = JLSparse(n_components=dimred)
        return jls.fit_transform(X).toarray()
    elif method == 'hvg':
        X = X.tocsc()
        disp = dispersion(X)
        highest_disp_idx = np.argsort(disp)[::-1][:dimred]
        return X[:, highest_disp_idx].toarray()
    else:
        sys.stderr.write('ERROR: Unknown method {}.'.format(method))
        exit(1)

def dispersion(X, eps=1e-10):
    mean = X.mean(0).A1
    
    X_nonzero = X[:, mean > eps]
    nonzero_mean = X_nonzero.mean(0).A1
    nonzero_var = (X_nonzero.multiply(X_nonzero)).mean(0).A1
    del X_nonzero
    
    nonzero_dispersion = (nonzero_var / nonzero_mean)

    dispersion = np.zeros(X.shape[1])
    dispersion[mean > eps] = nonzero_dispersion
    dispersion[mean <= eps] = float('-inf')
    
    return dispersion

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
