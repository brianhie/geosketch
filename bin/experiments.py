import numpy as np
import os
from scanorama import visualize
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import sys

from sketch import *
from utils import *

def experiment_efficiency_kmeans(X_dimred, cluster_labels,
                                 verbose=True):

    log('k-means clustering efficiency experiment...')
    
    cluster_labels = np.array(cluster_labels)
    k_c_e = {}
    kmeans_ks = [ 3, 5, 10, 20, 30, 50, 100 ]
    
    for kmeans_k in kmeans_ks:
        log('k = {}'.format(kmeans_k))

        km = KMeans(n_clusters=kmeans_k, n_jobs=10, verbose=0)
        km.fit(X_dimred)

        clusters = sorted(set(cluster_labels))
    
        log('Calculating cluster efficiencies for k = {}'
            .format(kmeans_k))

        cluster_to_efficiency = {}
        for cluster in clusters:
            efficiency = cluster_efficiency(
                cluster, cluster_labels, km.labels_
            )
            cluster_to_efficiency[cluster] = efficiency

        k_c_e[kmeans_k] = cluster_to_efficiency

    log('k-means clustering efficiency experiment done.')

    return k_c_e

# Clustering-based downsampling efficiency.
def cluster_efficiency(cluster, cluster_labels, auto_labels):
    assert(len(cluster_labels) == len(auto_labels))

    n_cluster_in_auto = []
    pct_cluster_in_auto = []

    autos = sorted(set(auto_labels))

    for auto in autos:
        auto_idx = auto_labels == auto
        n_auto = sum(auto_idx)
        in_auto = cluster_labels[auto_idx]
        n_in_auto = sum(in_auto == cluster)
        pct_in_auto = float(n_in_auto) / float(n_auto)

        n_cluster_in_auto.append(n_in_auto)
        pct_cluster_in_auto.append(pct_in_auto)

    # Weighted sum.
    wsum = np.dot(n_cluster_in_auto, pct_cluster_in_auto)

    # Compute clustering-based downsampling efficiency.
    # Divide weighted sum by number of samples in the cluster.
    return float(wsum) / float(sum(n_cluster_in_auto))

def experiment_srs(X_dimred, name, kmeans=True, visualize_orig=True,
                   downsample=True, n_downsample=100000,
                   gene_names=None, gene_expr=None, genes=None,
                   perplexity=500, kmeans_k=10):

    # Assign cells to clusters.

    if kmeans or \
       not os.path.isfile('data/cell_labels/{}.txt'.format(name)):
        log('K-means...')
        km = KMeans(n_clusters=kmeans_k, n_jobs=10, verbose=0)
        km.fit(X_dimred)
        np.savetxt('data/cell_labels/{}.txt'.format(name), km.labels_)
    
    cell_labels = (
        open('data/cell_labels/{}.txt'.format(name))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    # Visualize original data.
    
    if visualize_orig:
        log('Visualizing original...')
     
        if downsample:
            log('Visualization will downsample to {}...'
                .format(n_downsample))
            idx = np.random.choice(
                X_dimred.shape[0], size=n_downsample, replace=False
            )
        else:
            idx = range(X_dimred.shape[0])

        if not gene_names is None and \
           not gene_expr is None and \
           not genes is None:
            expr = gene_expr[idx, :]
     
        embedding = visualize(
            [ X_dimred[idx, :] ], cell_labels[idx],
            name + '_orig{}'.format(len(idx)), cell_types,
            gene_names=gene_names, gene_expr=expr, genes=genes,
            perplexity=perplexity, n_iter=400, image_suffix='.png'
        )
        np.savetxt('data/embedding_{}.txt'.format(name), embedding)

    # Downsample while preserving structure and visualize.

    Ns = [ 1000, 5000, 10000, 20000, 50000 ]

    for N in Ns:
        if N >= X_dimred.shape[0]:
            continue

        log('SRS {}...'.format(N))
        srs_idx = srs(X_dimred, N)
        log('Found {} entries'.format(len(set(srs_idx))))

        log('Visualizing sampled...')

        if not gene_names is None and \
           not gene_expr is None and \
           not genes is None:
            expr = gene_expr[srs_idx, :]

        visualize([ X_dimred[srs_idx, :] ], cell_labels[srs_idx],
                  name + '_srs{}'.format(N), cell_types,
                  gene_names=gene_names, gene_expr=expr, genes=genes,
                  perplexity=max(N/200, 50), n_iter=500,
                  size=max(int(30000/N), 1), image_suffix='.png')
