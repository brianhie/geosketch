from fbpca import pca
import math
import numpy as np
import os
import pandas as pd
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, scale

from artificial_density import kl_divergence
from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'artificial_volume'
METHOD = 'svd'
DIMRED = 3

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
        X_shrink = X_dimred / (10 ** i)
        #X_shrink[:, 0] = (X_shrink[:, 0] / (10 ** i))
        Xs.append(X_shrink + (translate * 2 * i))
        labels += list(np.zeros(X_dimred.shape[0]) + i)
        
    X_dimred = np.concatenate(Xs)
    cell_labels = np.array(labels, dtype=int)

    expected = np.array([ 1., 1. / 10, 1. / 100])
    expected = np.array(expected) / sum(expected)

    from geosketch import gs_gap, srs, kmeanspp, uniform

    sampling_fns = [
        uniform,
        gs_gap,
        kmeanspp,
        srs,
    ]
    sampling_fn_names = [
        'uniform',
        'gs_gap_N',
        'kmeans++',
        'srs',
    ]

    stats = pd.DataFrame(columns=[ 'name', 'kl_divergence', 'seed' ])
    n_seeds = 10
    
    for s_idx, sampling_fn in enumerate(sampling_fns):
        name = sampling_fn_names[s_idx]
        for seed in range(n_seeds):
            if name == 'gs_gap_N':
                samp_idx = sampling_fn(X_dimred, 3000, k=3000)
            else:
                samp_idx = sampling_fn(X_dimred, 3000)
            stats = stats.append({
                'name': name,
                'kl_divergence': kl_divergence(
                    cell_labels, samp_idx, expected
                ),
                'seed': seed,
            }, ignore_index=True)

    plt.figure()
    sns.barplot(x='name', y='kl_divergence', data=stats,
                order=sorted(sampling_fn_names),
                palette=[ '#377eb8', '#ff7f00', '#4daf4a', '#f781bf' ])
    plt.yscale('log')
    plt.savefig('artificial_volume.svg')
