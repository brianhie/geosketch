import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'zeisel'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/mouse_brain/zeisel/amygdala',
    'data/mouse_brain/zeisel/cerebellum',
    'data/mouse_brain/zeisel/cortex1',
    'data/mouse_brain/zeisel/cortex2',
    'data/mouse_brain/zeisel/cortex3',
    'data/mouse_brain/zeisel/enteric',
    'data/mouse_brain/zeisel/hippocampus',
    'data/mouse_brain/zeisel/hypothalamus',
    'data/mouse_brain/zeisel/medulla',
    'data/mouse_brain/zeisel/midbraindorsal',
    'data/mouse_brain/zeisel/midbrainventral',
    'data/mouse_brain/zeisel/olfactory',
    'data/mouse_brain/zeisel/pons',
    'data/mouse_brain/zeisel/spinalcord',
    'data/mouse_brain/zeisel/striatumdorsal',
    'data/mouse_brain/zeisel/striatumventral',
    'data/mouse_brain/zeisel/sympathetic',
    'data/mouse_brain/zeisel/thalamus',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    
    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    from ample import gs, uniform
    #samp_idx = gs(X_dimred, 20000, replace=False)
    samp_idx = uniform(X_dimred, 20000, replace=False)
    
    #from anndata import AnnData
    #import scanpy.api as sc
    #adata = AnnData(X=X_dimred[samp_idx, :])
    #sc.pp.neighbors(adata, use_rep='X')
    #sc.tl.louvain(adata, resolution=1.5, key_added='louvain')
    #
    #louv_labels = np.array(adata.obs['louvain'].tolist())
    #le = LabelEncoder().fit(louv_labels)
    #cell_labels = le.transform(louv_labels)
    #
    #np.savetxt('data/cell_labels/zeisel_louvain.txt', cell_labels)
    
    labels = (
        open('data/cell_labels/zeisel_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)
    
    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)

    exit()
    embedding = visualize(
        [ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
        NAMESPACE + '_uni{}'.format(len(samp_idx)),
        [ str(ct) for ct in sorted(set(cell_labels)) ],
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )
    exit()
    
    cell_labels = (
        open('data/cell_labels/zeisel_louvain.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    astro = set([ 32, 38, 40, ])
    oligo = set([ 2, 5, 12, 20, 23, 33, 37, ])
    focus = set([ 15, 36, 41 ])
    
    labels = []
    aob_labels = []
    for cl in cell_labels:
        if cl in focus:
            labels.append(0)
            aob_labels.append('both')
        elif cl in astro or cl in oligo:
            labels.append(1)
            if cl in astro:
                aob_labels.append('astro')
            else:
                aob_labels.append('oligo')
        else:
            labels.append(2)
            aob_labels.append('none')
    labels = np.array(labels)
    aob_labels = np.array(aob_labels)

    X = np.log1p(normalize(X[samp_idx, :]))

    from mouse_brain_astrocyte import astro_oligo_joint, astro_oligo_violin
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'astro', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'oligo', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'both', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'astro', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'oligo', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'both', NAMESPACE)
    
    astro_oligo_violin(X, genes, 'GJA1', aob_labels, NAMESPACE)
    astro_oligo_violin(X, genes, 'MBP', aob_labels, NAMESPACE)
    astro_oligo_violin(X, genes, 'PLP1', aob_labels, NAMESPACE)
    
    viz_genes = [
        #'GJA1', 'MBP', 'PLP1', 'TRF',
        #'CST3', 'CPE', 'FTH1', 'APOE', 'MT1', 'NDRG2', 'TSPAN7',
        #'PLP1', 'MAL', 'PTGDS', 'CLDN11', 'APOD', 'QDPR', 'MAG', 'ERMN',
        #'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        #'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        #'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    cell_labels = (
        open('data/cell_labels/zeisel_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    
    embedding = visualize(
        [ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
        NAMESPACE + '_astro{}'.format(len(samp_idx)),
        [ str(ct) for ct in sorted(set(cell_labels)) ],
        gene_names=viz_genes, gene_expr=X, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )
    
    #visualize_dropout(X, embedding, image_suffix='.png',
    #                  viz_prefix=NAMESPACE + '_dropout')
