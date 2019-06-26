import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
import sys

from experiments import *
from integration import harmony, integrate_sketch
from process import load_names
from utils import *

np.random.seed(0)

NAMESPACE = 'ica'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/ica/ica_bone_marrow_h5',
    'data/ica/ica_cord_blood_h5',
]

def entropy_test(X_dimred, ds_labels):
    ds_labels = np.array(ds_labels)
    #X_dimred = np.concatenate(datasets_dimred)
    embedding = None
    
    for k in range(2, 21):
        km = KMeans(n_clusters=k, n_jobs=-1, verbose=0)
        km.fit(X_dimred)

        if False and k % 5 == 0:
            embedding = visualize(
                datasets_dimred,
                km.labels_, NAMESPACE + '_km{}'.format(k),
                [ str(x) for x in range(k) ],
                embedding=embedding
            )
        
        print('k = {}, average normalized entropy = {}'
              .format(k, avg_norm_entropy(ds_labels, km.labels_)))

def avg_norm_entropy(ds_labels, cluster_labels):
    assert(len(ds_labels) == len(cluster_labels))
    
    clusters = sorted(set(cluster_labels))
    datasets = sorted(set(ds_labels))

    Hs = []
    for cluster in clusters:

        cluster_idx = cluster_labels == cluster
        ds_rep = ds_labels[cluster_idx]
        n_cluster = float(sum(cluster_idx))

        H = 0
        for ds in datasets:
            n_ds = float(sum(ds_rep == ds))
            if n_ds == 0: # 0 log 0 = 0
                continue
            H += (n_ds / n_cluster) * np.log(n_ds / n_cluster)
        H *= -1
        H /= np.log(len(datasets))

        Hs.append(H)
        
    return np.mean(Hs)

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= 500 ]
    X = X[gt_idx]

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    from geosketch import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '1000')
    
    for scale in [ 10, 25, 100 ]:
        N = int(X.shape[0] / scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + str(N))

    exit()
    
    datasets_full, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets_full, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    #log('Harmony (regular)...')
    #harmony_full = np.concatenate(harmony(datasets_dimred[:]))
    
    log('Harmony + GeoSketch...')
    harmony_sketch = np.concatenate(integrate_sketch(
        datasets_dimred[:], harmony,
    ))
    
    log('Scanorama + GeoSketch...')
    scanorama_sketch = np.concatenate(integrate_sketch(
        datasets_dimred[:], assemble, integration_fn_args={ 'knn': 50 },
    ))

    #log('Scanorama (regular)...')
    #scanorama_full = np.concatenate(assemble(
    #    datasets_dimred[:], knn=200, batch_size=1000
    #))
    
    log('Done integrating.')
    
    idx = np.random.choice(sum([ ds.shape[0] for ds in datasets ]),
                           size=20000, replace=False)

    integrations = [
        harmony_sketch,
        scanorama_sketch,
        #harmony_full,
        #scanorama_full,
    ]
    integration_names = [
        'harmony_sketch',
        'scanorama_sketch',
        #'harmony_full',
        #'scanorama_full'
    ]
    
    for integration, name in zip(integrations, integration_names):
        embedding = visualize(
            [ integration[idx] ], labels[idx], name,
            [ str(ct) for ct in sorted(set(labels)) ],
            perplexity=100, n_iter=500, image_suffix='.png',
            viz_cluster=False
        )
        
    for integration, name in zip(integrations, integration_names):
        adata = AnnData(X=integration[idx])
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        embedding = np.array(adata.obsm['X_umap'])
        embedding[embedding < -20] = -20
        visualize(
            None, labels[idx], name + '_umap',
            [ str(ct) for ct in sorted(set(labels)) ],
            image_suffix='.png', viz_cluster=False,
            embedding=embedding
        )
        
        print(name)
        entropy_test(embedding, labels[idx])

    exit()

    log('Scanorama integration geosketch')
    datasets_dimred, genes = integrate(
        datasets, genes_list, batch_size=1000, geosketch=True, dimred=20,
        geosketch_max=2000, knn=30, n_iter=20,
    )
    log('Done')

    embedding = visualize(
        [ np.concatenate(datasets_dimred)[idx] ], labels[idx],
        NAMESPACE + '_geosketch',
        [ str(ct) for ct in sorted(set(labels)) ],
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )

    entropy_test(embedding, labels[idx])

    datasets, genes_list, n_cells = load_names(data_names)
    log('Scanorama integration normal')
    datasets_dimred, genes = integrate(
        datasets, genes_list, batch_size=1000, dimred=20,
        knn=40
    )
    log('Done')
    
    embedding = visualize(
        [ np.concatenate(datasets_dimred)[idx] ], labels[idx],
        NAMESPACE + '_ds',
        [ str(ct) for ct in sorted(set(labels)) ],
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )
    
    entropy_test(embedding, labels[idx])
    
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    embedding = visualize(
        [ np.concatenate(datasets_dimred)[idx] ], labels[idx],
        NAMESPACE + '_uncorrected',
        [ str(ct) for ct in sorted(set(labels)) ],
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )
    
    entropy_test(embedding, labels[idx])

