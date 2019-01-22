from fbpca import pca
import numpy as np
import os
import pandas as pd
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'artificial_density'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/293t_jurkat/293t' ]

def kl_divergence(cell_labels, samp_idx, expected):
    cluster_labels = cell_labels[samp_idx]
    clusters = sorted(set(cell_labels))
    max_cluster = max(clusters)
    cluster_hist = np.zeros(max_cluster + 1)
    for c in range(max_cluster + 1):
        if c in clusters:
            cluster_hist[c] = np.sum(cluster_labels == c)
    cluster_hist /= np.sum(cluster_hist)
    return scipy.stats.entropy(cluster_hist, expected)

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
                samp_idx = sampling_fn(X_dimred, 200, k=200)
            else:
                samp_idx = sampling_fn(X_dimred, 200)
            stats = stats.append({
                'name': name,
                'kl_divergence': kl_divergence(
                    cell_labels, samp_idx, np.array([ 1./3, 1./3, 1./3])
                ),
                'seed': seed,
            }, ignore_index=True)

    plt.figure()
    sns.barplot(x='name', y='kl_divergence', data=stats,
                order=sorted(sampling_fn_names),
                palette=[ '#377eb8', '#ff7f00', '#4daf4a', '#f781bf' ])
    plt.savefig('artificial_density.svg')
