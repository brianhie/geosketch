from anndata import AnnData
import numpy as np
import os
from scanorama import *
import scanpy.api as sc
import scipy.stats
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from subprocess import Popen
import sys
from time import time

from process import load_names
from save_mtx import save_mtx
from supervised import adjusted_mutual_info_score
from ample import *
from utils import *

# Clustering-based downsampling efficiency.
def cluster_efficiency(cluster_labels, auto_labels):
    assert(len(cluster_labels) == len(auto_labels))

    clusters = sorted(set(cluster_labels))
    autos = sorted(set(auto_labels))

    # Assign indices to clusters and autos.
    cluster_idx = {
        cluster: i for i, cluster in enumerate(clusters)
    }
    auto_idx = {
        auto: j for j, auto in enumerate(autos)
    }

    # Count cluster-auto pairs.
    table = np.zeros((len(cluster_idx), len(auto_idx)))
    for cluster, auto in zip(cluster_labels, auto_labels):
        i = cluster_idx[cluster]
        j = auto_idx[auto]
        table[i, j] += 1

    # Map clusters to efficiencies.
    cluster_to_efficiency = {}
    for i, cluster in enumerate(clusters):

        n_cluster_in_auto = []
        pct_cluster_in_auto = []
        for j, auto in enumerate(autos):
            n_auto = sum(table[:, j])
            n_in_auto = table[i, j]
            pct_in_auto = float(n_in_auto) / float(n_auto)
    
            n_cluster_in_auto.append(n_in_auto)
            pct_cluster_in_auto.append(pct_in_auto)

        wsum = np.dot(n_cluster_in_auto, pct_cluster_in_auto)
        assert(sum(n_cluster_in_auto) == sum(table[i, :]))
        cluster_to_efficiency[cluster] = (
            float(wsum) / float(sum(n_cluster_in_auto))
        )

    return cluster_to_efficiency

def average_cluster_efficiency(cluster_labels, auto_labels):
    c_to_e = cluster_efficiency(cluster_labels, auto_labels)
    return np.mean([ c_to_e[c] for c in c_to_e ])

def report_cluster_counts(cluster_labels):
    clusters = sorted(set(cluster_labels))

    for cluster in clusters:
        n_cluster = sum(cluster_labels == cluster)
        print('Cluster {} has {} cells'.
              format(cluster, n_cluster))

def experiment_kmeanspp(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'kmeanspp'
    experiment(kmeanspp, X_dimred, name, **kwargs)
    
def experiment_srs(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'srs'
    experiment(srs, X_dimred, name, **kwargs)

def experiment_gs(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'gs'
    experiment(gs, X_dimred, name, **kwargs)

def experiment_uni(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'uni'
    experiment(uniform, X_dimred, name, **kwargs)
    
def experiment(sampling_fn, X_dimred, name, cell_labels=None,
               kmeans=True, visualize_orig=True,
               downsample=True, n_downsample=100000,
               gene_names=None, gene_expr=None, genes=None,
               perplexity=500, kmeans_k=10, sample_type=''):

    # Assign cells to clusters.

    if kmeans or \
       not os.path.isfile('data/cell_labels/{}.txt'.format(name)):
        log('K-means...')
        km = KMeans(n_clusters=kmeans_k, n_jobs=10, verbose=0)
        km.fit(X_dimred)
        np.savetxt('data/cell_labels/{}.txt'.format(name), km.labels_)

    if cell_labels is None:
        cell_labels = (
            open('data/cell_labels/{}.txt'.format(name))
            .read().rstrip().split()
        )
        le = LabelEncoder().fit(cell_labels)
        cell_labels = le.transform(cell_labels)
        cell_types = le.classes_
    else:
        cell_types = [ str(ct) for ct in sorted(set(cell_labels)) ]

    # Visualize original data.
    
    if visualize_orig:
        log('Visualizing original...')
     
        if downsample and X_dimred.shape[0] > n_downsample:
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
        else:
            expr = None
     
        embedding = visualize(
            [ X_dimred[idx, :] ], cell_labels[idx],
            name + '_orig{}'.format(len(idx)), cell_types,
            gene_names=gene_names, gene_expr=expr, genes=genes,
            perplexity=perplexity, n_iter=500, image_suffix='.png'
        )
        np.savetxt('data/embedding_{}.txt'.format(name), embedding)

    # Downsample while preserving structure and visualize.

    Ns = [ 1000, 5000, 10000, 20000, 50000 ]

    for N in Ns:
        if N >= X_dimred.shape[0]:
            continue

        log('Sampling {}...'.format(N))
        samp_idx = sampling_fn(X_dimred, N)
        log('Found {} entries'.format(len(set(samp_idx))))

        log('Visualizing sampled...')

        if not gene_names is None and \
           not gene_expr is None and \
           not genes is None:
            expr = gene_expr[samp_idx, :]
        else:
            expr = None

        visualize([ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
                  name + '_{}{}'.format(sample_type, N), cell_types,
                  gene_names=gene_names, gene_expr=expr, genes=genes,
                  perplexity=max(N/200, 50), n_iter=500,
                  size=max(int(30000/N), 1), image_suffix='.png')

        report_cluster_counts(cell_labels[samp_idx])


def normalized_entropy(counts):
    k = len(counts)
    if k <= 1:
        return 1
    
    n_samples = sum(counts)

    H = -sum([ (counts[i] / n_samples) * np.log(counts[i] / n_samples)
               for i in range(k) if counts[i] > 0 ])
    
    return H / np.log(k)

def err_exit(param_name):
    sys.stderr.write('Needs `{}\' param.\n'.format(param_name))
    exit(1)
    
def experiments(X_dimred, name, n_seeds=10, **kwargs):

    columns = [
        'name', 'sampling_fn', 'replace', 'N', 'seed', 'time'
    ]
    
    if 'rare' in kwargs and kwargs['rare']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        if not 'rare_label' in kwargs:
            err_exit('rare_label')
        columns.append('rare')
            
    if 'entropy' in kwargs and kwargs['entropy']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('entropy')

    if 'kl_divergence' in kwargs and kwargs['kl_divergence']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        if not 'expected' in kwargs:
            err_exit('expected')
        columns.append('kl_divergence')

    if 'max_min_dist' in kwargs and kwargs['max_min_dist']:
        columns.append('max_min_dist')
        
    if 'kmeans_ami' in kwargs and kwargs['kmeans_ami']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('kmeans_ami')
        columns.append('kmeans_bami')
        
    if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('louvain_ami')
        columns.append('louvain_bami')
        
    of = open('target/experiments/{}.txt.1'.format(name), 'a')
    of.write('\t'.join(columns) + '\n')
    
    Ns = [ 100, 500, 1000, 5000, 10000, 20000 ]

    sampling_fns = [
        uniform,
        gs_grid,
        gs_gap,
        srs,
        louvain1,
        louvain3,
        kmeans,
        kmeansppp,
        kmeanspp,
    ]
    
    sampling_fn_names = [
        'uniform',
        'gs_grid',
        'gs_gap',
        'srs',
        'louvain1',
        'louvain3',
        'kmeans',
        'kmeans+++',
        'kmeans++',
    ]

    not_replace = set([ 'kmeans++', 'dropClust' ])

    assert(len(sampling_fns) == len(sampling_fn_names))

    for s_idx, sampling_fn in enumerate(sampling_fns):
        
        if sampling_fn_names[s_idx] == 'dropClust':
            dropclust_preprocess(X_dimred, name)

        for replace in [ True, False ]:
            if replace and sampling_fn_names[s_idx] in not_replace:
                continue

            counts_means, counts_sems = [], []

            for N in Ns:
                if N > X_dimred.shape[0]:
                    continue
                log('N = {}...'.format(N))
            
                counts = []
                
                for seed in range(n_seeds):

                    if sampling_fn_names[s_idx] == 'dropClust':
                        log('Sampling dropClust...')
                        t0 = time()
                        samp_idx = dropclust_sample('data/' + name, N, seed=seed)
                        t1 = time()
                        log('Sampling dropClust done.')
                    elif sampling_fn_names[s_idx] == 'gs_gap_N':
                        log('Sampling gs_gap_N...')
                        t0 = time()
                        samp_idx = sampling_fn(X_dimred, N, k=N, seed=seed,
                                               replace=replace)
                        t1 = time()
                        log('Sampling gs_gap_N done.')
                    else:
                        log('Sampling {}...'.format(sampling_fn_names[s_idx]))
                        t0 = time()
                        samp_idx = sampling_fn(X_dimred, N, seed=seed,
                                               replace=replace)
                        t1 = time()
                        log('Sampling {} done.'.format(sampling_fn_names[s_idx]))

                    kwargs['sampling_fn'] = sampling_fn_names[s_idx]
                    kwargs['replace'] = replace
                    kwargs['N'] = N
                    kwargs['seed'] = seed
                    kwargs['time'] = t1 - t0
                    
                    experiment_stats(of, X_dimred, samp_idx, name, **kwargs)

    of.close()

def experiment_stats(of, X_dimred, samp_idx, name, **kwargs):
    stats = [
        name,
        kwargs['sampling_fn'],
        kwargs['replace'],
        kwargs['N'],
        kwargs['seed'],
        kwargs['time'],
    ]

    if 'rare' in kwargs and kwargs['rare']:
        cell_labels = kwargs['cell_labels']
        rare_label = kwargs['rare_label']
        cluster_labels = cell_labels[samp_idx]
        stats.append(sum(cluster_labels == rare_label))

    if 'entropy' in kwargs and kwargs['entropy']:
        cell_labels = kwargs['cell_labels']
        cluster_labels = cell_labels[samp_idx]
        clusters = sorted(set(cell_labels))
        max_cluster = max(clusters)
        cluster_hist = np.zeros(max_cluster + 1)
        for c in range(max_cluster + 1):
            if c in clusters:
                cluster_hist[c] = np.sum(cluster_labels == c)
        stats.append(normalized_entropy(cluster_hist))

    if 'kl_divergence' in kwargs and kwargs['kl_divergence']:
        cell_labels = kwargs['cell_labels']
        expected = kwargs['expected']
        cluster_labels = cell_labels[samp_idx]
        clusters = sorted(set(cell_labels))
        max_cluster = max(clusters)
        cluster_hist = np.zeros(max_cluster + 1)
        for c in range(max_cluster + 1):
            if c in clusters:
                cluster_hist[c] = np.sum(cluster_labels == c)
        cluster_hist /= np.sum(cluster_hist)
        stats.append(scipy.stats.entropy(expected, cluster_hist))

    if 'max_min_dist' in kwargs and kwargs['max_min_dist']:
        dist = pairwise_distances(
            X_dimred[samp_idx, :], X_dimred, n_jobs=-1
        )
        stats.append(dist.min(0).max())

    if 'kmeans_ami' in kwargs and kwargs['kmeans_ami']:
        cell_labels = kwargs['cell_labels']
        
        k = len(set(cell_labels))
        km = KMeans(n_clusters=k, n_init=1, random_state=kwargs['seed'])
        km.fit(X_dimred[samp_idx, :])

        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], km.labels_)
                
        ami = adjusted_mutual_info_score(cell_labels, full_labels)
        bami = adjusted_mutual_info_score(
            cell_labels, full_labels, dist='balanced'
        )
        stats.append(ami)
        stats.append(bami)

    if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
        cell_labels = kwargs['cell_labels']
        
        adata = AnnData(X=X_dimred[samp_idx, :])
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.louvain(adata, resolution=1., key_added='louvain')
        louv_labels = np.array(adata.obs['louvain'].tolist())

        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], louv_labels)

        ami = adjusted_mutual_info_score(cell_labels, full_labels)
        bami = adjusted_mutual_info_score(
            cell_labels, full_labels, dist='balanced'
        )
        stats.append(ami)
        stats.append(bami)
        
    of.write('\t'.join([ str(stat) for stat in stats ]) + '\n')
    of.flush()
    
def seurat_cluster(name):
    rcode = Popen('Rscript R/seurat.R {0} > {0}.log 2>&1'
                  .format(name), shell=True).wait()
    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)

    labels = []
    with open(name + '_labels.txt') as f:
        f.readline()
        for line in f:
            labels.append(line.rstrip().split()[1])
    return labels

def experiment_seurat_ari(data_names, namespace):
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    X_dimred = reduce_dimensionality(normalize(X))

    name = 'data/{}'.format(namespace)
    Ns = [ 500, 1000, 2000, 5000, 10000 ]
    
    if not os.path.isfile('{}/matrix.mtx'.format(name)):
        save_mtx(name, csr_matrix(X), genes)
    log('Seurat clustering full dataset...')
    cluster_labels_full = seurat_cluster(name)
    log('Seurat clustering done.')

    for N in Ns:
        gs_idx = gs(X_dimred, N)
        save_mtx(name + '/gs{}'.format(N), csr_matrix(X[gs_idx, :]),
                 genes)
        log('Seurat clustering GS N = {}...'.format(N))
        seurat_labels = seurat_cluster(name + '/gs{}'.format(N))
        log('Seurat clustering GS N = {} done.'.format(N))
        cluster_labels = label_approx(
            X_dimred, X_dimred[gs_idx], seurat_labels
        )
        log('N = {}, GS ARI = {}'.format(
            N, adjusted_rand_score(cluster_labels_full, cluster_labels)
        ))
        
        uni_idx = uniform(X_dimred, N)
        save_mtx(name + '/uni{}'.format(N), csr_matrix(X[uni_idx, :]),
                 genes)
        log('Seurat clustering uniform N = {}...'.format(N))
        seurat_labels = seurat_cluster(name + '/uni{}'.format(N))
        log('Seurat clustering uniform N = {} done.'.format(N))
        cluster_labels = label_approx(
            X_dimred, X_dimred[uni_idx], seurat_labels
        )
        log('N = {}, Uniform ARI = {}'.format(
            N, adjusted_rand_score(cluster_labels_full, cluster_labels)
        ))

def experiment_kmeans_ce(X, name, cell_labels, n_seeds=10, N=None):
    
    sampling_fns = [
        uniform,
        gs,
        gs_gap,
        srs,
        louvain1,
        louvain3,
        kmeanspp,
        kmeansppp,
    ]
    
    sampling_fn_names = [
        'uniform',
        'gs_grid',
        'gs_gap',
        'srs',
        'louvain1',
        'louvain3',
        'kmeans++',
        'kmeans+++',
    ]

    of = open('target/experiments/kmeans_ce/{}.txt'.format(name), 'a')

    columns = [ 'name', 'sampling_fn', 'replace', 'N', 'k', 'seed',
                'clust_eff' ]
    of.write('\t'.join(columns) + '\n')

    if N is None:
        N = int(X.shape[0] / 100)

    ks = [ 5, 10, 20, 50, int(np.sqrt(X.shape[0])) ]

    for s_idx, sampling_fn in enumerate(sampling_fns):

        for k in ks:
            
            if k > N:
                continue

            for seed in range(n_seeds):
                
                samp_idx = sampling_fn(X, N, seed=seed)
    
                km = KMeans(n_clusters=k, n_init=1, random_state=seed)
                km.fit(X[samp_idx, :])

                full_labels = label_approx(X, X[samp_idx, :], km.labels_)
                
                avg_ce = average_cluster_efficiency(
                    cell_labels, full_labels
                )
    
                stats = [
                    name, sampling_fn_names[s_idx], False, N, k, seed, avg_ce
                ]
                
                of.write('\t'.join([ str(stat) for stat in stats ]) + '\n')
                of.flush()

    of.close()

def experiment_louvain_ce(X, name, cell_labels, n_seeds=10, N=None):
    sampling_fns = [
        uniform,
        gs,
        gs_gap,
        srs,
        louvain1,
        louvain3,
        kmeanspp,
        kmeansppp,
    ]
    
    sampling_fn_names = [
        'uniform',
        'gs_grid',
        'gs_gap',
        'srs',
        'louvain1',
        'louvain3',
        'kmeans++',
        'kmeans+++',
    ]

    of = open('target/experiments/louvain_ce/{}.txt'.format(name), 'a')

    columns = [ 'name', 'sampling_fn', 'replace', 'N', 'resolution', 'seed',
                'k', 'clust_eff' ]
    of.write('\t'.join(columns) + '\n')
    
    if N is None:
        N = int(X.shape[0] / 100)

    resolutions = [ 0.1, 0.5, 1, 2, 3, 5 ]

    for s_idx, sampling_fn in enumerate(sampling_fns):

        for seed in range(n_seeds):
                
            samp_idx = sampling_fn(X, N, seed=seed)
    
            for resolution in resolutions:
            
                adata = AnnData(X=X[samp_idx, :])
                sc.pp.neighbors(adata, use_rep='X')
                sc.tl.louvain(adata, resolution=resolution, key_added='louvain')
                louv_labels = np.array(adata.obs['louvain'].tolist())

                full_labels = label_approx(X, X[samp_idx, :], louv_labels)
                
                avg_ce = average_cluster_efficiency(
                    cell_labels, full_labels
                )
    
                stats = [
                    name, sampling_fn_names[s_idx], False, N, resolution, seed,
                    max(louv_labels), avg_ce
                ]
                
                of.write('\t'.join([ str(stat) for stat in stats ]) + '\n')
                of.flush()

    of.close()
    
def save_sketch(X, samp_idx, genes, namespace):
    name = 'data/{}'.format(namespace)
    mkdir_p(name)
    save_mtx(name, csr_matrix(X[samp_idx, :]), genes)
