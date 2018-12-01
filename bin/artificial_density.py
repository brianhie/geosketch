from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'artificial_density'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/293t_jurkat/293t' ]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    
    k = DIMRED
    U, s, Vt = pca(normalize(X), k=k)
    X_dimred = U[:, :k] * s[:k]

    Xs = []
    labels = []
    translate = X_dimred.max(0)
    for i in range(3):
        rand_idx = np.random.choice(
            X.shape[0], size=int(X.shape[0] / (10 ** i)), replace=False
        )
        Xs.append(X_dimred[rand_idx, :] + (translate * 2 * i))
        labels += list(np.zeros(len(rand_idx)) + i)

        print(int(X.shape[0] / (10 ** i)))
        
    X_dimred = np.concatenate(Xs)
    cell_labels = np.array(labels, dtype=int)

    from ample import gs, gs_gap, srs, srs_positive
    samp_idx = gs_gap(X_dimred, 3000, replace=True)
    report_cluster_counts(cell_labels[samp_idx])
    samp_idx = srs(X_dimred, 3000, replace=True)
    report_cluster_counts(cell_labels[samp_idx])
    samp_idx = srs_positive(X_dimred, 3000, replace=True)
    report_cluster_counts(cell_labels[samp_idx])
    exit()
    
    experiments(
        X_dimred, NAMESPACE,
        rare=True, cell_labels=cell_labels, rare_label=2,
        entropy=True,
        kl_divergence=True, expected=np.array([ 1./3, 1./3, 1./3]),
        max_min_dist=True
    )
