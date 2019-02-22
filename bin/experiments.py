from anndata import AnnData
import math
import numpy as np
import os
import pandas as pd
from scanorama import *
import scanpy.api as sc
import scipy.stats
from scipy.sparse import vstack
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from subprocess import Popen
import sys
from time import time

from process import load_names
from save_mtx import save_mtx
from supervised import adjusted_mutual_info_score
from supervised import normalized_mutual_info_score
from geosketch import *
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
    #from geosketch import srs
    kwargs['sample_type'] = 'srs'
    experiment(srs, X_dimred, name, **kwargs)

def experiment_gs(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'gs'
    experiment(gs, X_dimred, name, **kwargs)

def experiment_uni(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'uni'
    experiment(uniform, X_dimred, name, **kwargs)
    
def experiment_louvain(X_dimred, name, **kwargs):
    kwargs['sample_type'] = 'louvain'
    experiment(louvain1, X_dimred, name, **kwargs)
    
def experiment(sampling_fn, X_dimred, name, cell_labels=None,
               N_only=None, kmeans=True, visualize_orig=True,
               downsample=True, n_downsample=100000,
               gene_names=None, gene_expr=None, genes=None,
               perplexity=500, kmeans_k=10, sample_type='',
               viz_type='tsne'):

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

    if N_only is None:
        Ns = [ 1000, 5000, 10000, 20000, 50000 ]
    else:
        Ns = [ N_only ]

    for N in Ns:
        if N >= X_dimred.shape[0]:
            continue

        log('Sampling {}...'.format(N))
        if sample_type == 'gs':
            samp_idx = sampling_fn(X_dimred, N, k=N)
        else:
            samp_idx = sampling_fn(X_dimred, N)
        log('Found {} entries'.format(len(set(samp_idx))))

        log('Visualizing sampled...')

        if not gene_names is None and \
           not gene_expr is None and \
           not genes is None:
            expr = gene_expr[samp_idx, :]
        else:
            expr = None

        if viz_type == 'umap':
            adata = AnnData(X=X_dimred[samp_idx, :])
            sc.pp.neighbors(adata, use_rep='X')
            sc.tl.umap(adata, min_dist=0.5)
            embedding = np.array(adata.obsm['X_umap'])
            embedding[embedding < -20] = -20
            embedding[embedding > 20] = -20
            visualize(None, cell_labels[samp_idx],
                      name + '_umap_{}{}'.format(sample_type, N), cell_types,
                      embedding=embedding,
                      gene_names=gene_names, gene_expr=expr, genes=genes,
                      size=max(int(30000/N), 5), image_suffix='.png')
        else:
            visualize([ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
                      name + '_{}{}'.format(sample_type, N), cell_types,
                      gene_names=gene_names, gene_expr=expr, genes=genes,
                      #perplexity=5, n_iter=500,
                      perplexity=max(N/200, 50), n_iter=500,
                      size=max(int(30000/N), 5), image_suffix='.png')

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

def load_cached_idx(name, dirname, N, seed, dtype=int):
    cache_fname = 'data/cache/{}/{}_{}_{}.txt'.format(dirname, name, N, seed)
    if os.path.isfile(cache_fname):
        with open(cache_fname, 'r') as f:
            return [ dtype(x) for x in f.read().rstrip().split() ]
    
def save_cached_idx(samp_idx, name, dirname, N, seed, dtype=int):
    cache_fname = 'data/cache/{}/{}_{}_{}.txt'.format(dirname, name, N, seed)
    with open(cache_fname, 'w') as of:
        for si in samp_idx:
            of.write(str(dtype(si)) + '\n')
        
def experiments(X_dimred, name, n_seeds=10, use_cache=True, **kwargs):

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
        columns.append('9999_min_dist')
        columns.append('999_min_dist')
        columns.append('99_min_dist')
        columns.append('95_min_dist')
        
    if 'kmeans_nmi' in kwargs and kwargs['kmeans_nmi']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('kmeans_nmi')
        columns.append('kmeans_bnmi')
        
    if 'spectral_nmi' in kwargs and kwargs['spectral_nmi']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('spectral_nmi')
        columns.append('spectral_bnmi')
        
    if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('louvain_ami')
        columns.append('louvain_bami')
        columns.append('louvain_n_clusters')
        
    if 'sub_labels' in kwargs:
        columns.append('n_subcluster')

    if 'cell_exp_ratio' in kwargs and kwargs['cell_exp_ratio']:
        if not 'cell_labels' in kwargs:
            err_exit('cell_labels')
        columns.append('cell_exp_ratio')
        
    if use_cache:
        mkdir_p('data/cache/{}'.format(name))

    mkdir_p('target/experiments')
        
    of = open('target/experiments/{}.txt.20'.format(name), 'a')
    of.write('\t'.join(columns) + '\n')

    if 'Ns' in kwargs and kwargs['Ns'] is not None:
        Ns = kwargs['Ns']
    else:
        Ns = []
        for scale in [ 0.02, 0.04, 0.06, 0.08, 0.1 ]:
            Ns.append(int(scale * X_dimred.shape[0]))
        Ns = np.array(Ns)
        Ns = [ int(N) for N in Ns ]

    sampling_fns = [
        uniform,
        #gs_gap,
        gs_gap,
        kmeanspp,
        srs,
        #pc_pick,
        #srs_positive,
        #gs_grid,
        #louvain1,
        #louvain3,
        #kmeans,
        #kmeansppp,
    ]
    
    sampling_fn_names = [
        'uniform',
        #'gs_gap',
        'gs_gap_N',
        'kmeans++',
        'srs',
        #'pc_pick',
        #'srs_positive',
        #'gs_grid',
        #'louvain1',
        #'louvain3',
        #'kmeans',
        #'kmeans+++',
    ]

    not_replace = set([ 'kmeans++', 'dropClust' ])

    assert(len(sampling_fns) == len(sampling_fn_names))

    for s_idx, sampling_fn in enumerate(sampling_fns):
        
        if sampling_fn_names[s_idx] == 'dropClust':
            dropclust_preprocess(X_dimred, name)

        for replace in [ False ]:#[ True, False ]:
            if replace and sampling_fn_names[s_idx] in not_replace:
                continue

            counts_means, counts_sems = [], []

            for N in Ns:
                if N > X_dimred.shape[0]:
                    continue
                log('N = {}...'.format(N))
            
                counts = []
                
                for seed in range(10, 10 + n_seeds):

                    if sampling_fn_names[s_idx] == 'dropClust':
                        log('Sampling dropClust...')
                        t0 = time()
                        samp_idx = dropclust_sample('data/' + name, N, seed=seed)
                        t1 = time()
                        log('Sampling dropClust done.')
                    elif sampling_fn_names[s_idx] == 'gs_gap_N':
                        log('Sampling gs_gap_N...')
                        if use_cache:
                            samp_idx = load_cached_idx(sampling_fn_names[s_idx],
                                                       name, N, seed)
                        else:
                            samp_idx = None
                        
                        if samp_idx is None:
                            t0 = time()
                            samp_idx = sampling_fn(X_dimred, N, k=N, seed=seed,
                                                   replace=replace)
                            t1 = time()
                            log('Sampling gs_gap_N done.')
                            if use_cache:
                                save_cached_idx(samp_idx, sampling_fn_names[s_idx],
                                                name, N, seed)
                        else:
                            t1 = t0 = 0
                    elif sampling_fn_names[s_idx] == 'gs_gap_2N':
                        log('Sampling gs_gap_2N...')
                        t0 = time()
                        samp_idx = sampling_fn(X_dimred, N, k=2*N, seed=seed,
                                               replace=replace)
                        t1 = time()
                        log('Sampling gs_gap_2N done.')
                    elif sampling_fn_names[s_idx] == 'gs_gap_k':
                        log('Sampling gs_gap_k...')
                        t0 = time()
                        samp_idx = sampling_fn(X_dimred, 20000, k=N, seed=seed,
                                               replace=replace)
                        t1 = time()
                        log('Sampling gs_gap_k done.')
                    else:
                        log('Sampling {}...'.format(sampling_fn_names[s_idx]))

                        if use_cache:
                            samp_idx = load_cached_idx(sampling_fn_names[s_idx],
                                                       name, N, seed)
                        else:
                            samp_idx = None

                        if samp_idx is None:
                        
                            t0 = time()
                            samp_idx = sampling_fn(X_dimred, N, seed=seed,
                                                   replace=replace)
                            t1 = time()
                            log('Sampling {} done.'.format(sampling_fn_names[s_idx]))

                            if use_cache:
                                save_cached_idx(samp_idx, sampling_fn_names[s_idx],
                                                name, N, seed)

                        else:
                            t1 = t0 = 0

                    kwargs['sampling_fn'] = sampling_fn_names[s_idx]
                    kwargs['replace'] = replace
                    kwargs['N'] = N
                    kwargs['seed'] = seed
                    kwargs['time'] = t1 - t0
                    kwargs['use_cache'] = True
                    
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
        stats.append(scipy.stats.entropy(cluster_hist, expected))

    if 'max_min_dist' in kwargs and kwargs['max_min_dist']:
        N = kwargs['N']
        seed = kwargs['seed']
        use_cache = kwargs['use_cache']
        
        if use_cache:
            min_dist = load_cached_idx('min_dist_' + kwargs['sampling_fn'],
                                       name, N, seed, dtype=float)
        else:
            min_dist = []
    
        if min_dist is None or len(min_dist) == 0:
            min_dist = []
            batch_size = 20000
            for batch in range(0, X_dimred.shape[0], batch_size):
                dist = pairwise_distances(
                    X_dimred[samp_idx, :], X_dimred[batch:batch + batch_size],
                    n_jobs=-1
                )
                min_dist += list(dist.min(0))
                del dist
            if use_cache:
                save_cached_idx(min_dist, 'min_dist_' + kwargs['sampling_fn'],
                                name, N, seed, dtype=float)

        #plt.figure()
        #sns.violinplot(data=[ min_dist ], scale='width', cut=0)
        #sns.stripplot(data=[ min_dist ], jitter=True, color='black', size=1)
        #plt.savefig('max_min_dist.png')
        #exit()

        stats.append(max(min_dist))
        stats.append(np.quantile(min_dist, 0.9999))
        stats.append(np.quantile(min_dist, 0.999))
        stats.append(np.percentile(min_dist, 99))
        stats.append(np.percentile(min_dist, 95))

    if 'kmeans_nmi' in kwargs and kwargs['kmeans_nmi']:
        cell_labels = kwargs['cell_labels']
        
        k = len(set(cell_labels))
        km = KMeans(n_clusters=k, n_init=1, random_state=kwargs['seed'])
        km.fit(X_dimred[samp_idx, :])

        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], km.labels_)
                
        nmi = normalized_mutual_info_score(cell_labels, full_labels)
        bnmi = normalized_mutual_info_score(
            cell_labels, full_labels, dist='balanced'
        )
        stats.append(nmi)
        stats.append(bnmi)
        
    if 'spectral_nmi' in kwargs and kwargs['spectral_nmi']:
        cell_labels = kwargs['cell_labels']
        
        k = len(set(cell_labels))
        spect = SpectralClustering(n_clusters=k*2,
                                   assign_labels='discretize',
                                   random_state=kwargs['seed'], n_jobs=-1)
        spect.fit(X_dimred[samp_idx, :])

        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :],
                                   spect.labels_, k=5)
                
        bnmi = normalized_mutual_info_score(
            cell_labels, full_labels, dist='balanced'
        )
        nmi = normalized_mutual_info_score(cell_labels, full_labels)
        stats.append(nmi)
        stats.append(bnmi)
        
    if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
        cell_labels = kwargs['cell_labels']

        adata = AnnData(X=X_dimred[samp_idx, :])
        sc.pp.neighbors(adata, use_rep='X')
        
        amis = []
        bamis = []
        
        for r in [ 0.5, 1., 2. ]:
            sc.tl.louvain(adata, resolution=r, key_added='louvain')
            louv_labels = np.array(adata.obs['louvain'].tolist())

            full_labels = label_approx(X_dimred, X_dimred[samp_idx, :],
                                       louv_labels, k=5)

            ami = adjusted_mutual_info_score(cell_labels, full_labels)
            bami = adjusted_mutual_info_score(
                cell_labels, full_labels, dist='balanced'
            )
            amis.append(ami)
            bamis.append(bami)
            
        stats.append(max(amis))
        stats.append(max(bamis))
        stats.append(len(set(louv_labels)))
        
    if 'sub_labels' in kwargs:
        sub_labels = kwargs['sub_labels']
        stats.append(len(set(sub_labels[samp_idx])))
        
    if 'cell_exp_ratio' in kwargs and kwargs['cell_exp_ratio']:
        cell_labels = kwargs['cell_labels']
        n_samples = float(X_dimred.shape[0])
        N = float(kwargs['N'])
        
        samp_idx_set = set(samp_idx)
        cell_count = {}
        samp_count = {}
        for idx, cell_label in enumerate(cell_labels):
            if cell_label not in cell_count:
                cell_count[cell_label] = 0
                samp_count[cell_label] = 0
            cell_count[cell_label] += 1
            if idx in samp_idx_set:
                samp_count[cell_label] += 1
                
        cell_labels_sorted = [
            x[1] for x in sorted([
                (cell_count[cell_label], cell_label)
                for cell_label in cell_count ])
        ]
        n_include = math.ceil(float(len(cell_count)) / 2.)
        include_labels = set(cell_labels_sorted[:n_include])

        cluster_ratios = []
        for cell_label in cell_count:
            if cell_label not in include_labels:
                continue
            observed = float(samp_count[cell_label])
            expected = float(cell_count[cell_label]) * (N / n_samples)
            cluster_ratios.append(observed / expected)                
                
        stats.append(scipy.stats.mstats.gmean(cluster_ratios))
        
    of.write('\t'.join([ str(stat) for stat in stats ]) + '\n')
    of.flush()

def plot_rare(X_dimred, cell_labels, rare_label, namespace,
              use_cache=True, n_seeds=10):
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

    N = int(0.02 * X_dimred.shape[0])
    
    for s_idx, sampling_fn in enumerate(sampling_fns):
        name = sampling_fn_names[s_idx]
        for seed in range(10, 10 + n_seeds):
            if use_cache:
                samp_idx = load_cached_idx(sampling_fn_names[s_idx],
                                           name, N, seed)
            else:
                samp_idx = None
            if samp_idx is None:
                if name == 'gs_gap_N':
                    samp_idx = sampling_fn(X_dimred, N, k=N)
                else:
                    samp_idx = sampling_fn(X_dimred, N)
            cluster_labels = cell_labels[samp_idx]
            stats = stats.append({
                'name': name,
                'rare': sum(cluster_labels == rare_label),
                'seed': seed,
            }, ignore_index=True)

    plt.figure()
    sns.barplot(x='name', y='rare', data=stats,
                order=sorted(sampling_fn_names),
                palette=[ '#377eb8', '#ff7f00', '#4daf4a', '#f781bf' ])
    plt.savefig('rare_barplot_{}.svg'.format(namespace))
    
    
def save_sketch(X, samp_idx, genes, namespace):
    name = 'data/{}'.format(namespace)
    mkdir_p(name)
    save_mtx(name, csr_matrix(X[samp_idx, :]), genes)
