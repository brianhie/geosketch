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
    
    labels = np.array(
        open('data/cell_labels/zeisel_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

    experiments(
        X_dimred, NAMESPACE, n_seeds=4,
        cell_labels=cell_labels,
        louvain_ami=True, spectral_nmi=True,
        rare=True,
        rare_label=le.transform(['Ependymal'])[0],
    )
    exit()

    visualize(
        None, cell_labels,
        NAMESPACE + '_tsne_full',
        [ str(ct) for ct in sorted(set(cell_labels)) ],
        embedding=np.loadtxt('data/embedding/embedding_zeisel_tsne.txt'),
        image_suffix='.png',        
    )
    visualize(
        None, cell_labels,
        NAMESPACE + '_umap_full',
        [ str(ct) for ct in sorted(set(cell_labels)) ],
        embedding=np.loadtxt('data/embedding/embedding_zeisel_umap.txt'),
        image_suffix='.png',        
    )
    
    report_cluster_counts(labels)
    
    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)

    experiment_gs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        #gene_names=viz_genes, genes=genes,
        #gene_expr=vstack(datasets),
        viz_type='umap', N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_uni(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        #gene_names=viz_genes, genes=genes,
        #gene_expr=vstack(datasets),
        viz_type='umap', N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_srs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        #gene_names=viz_genes, genes=genes,
        #gene_expr=vstack(datasets),
        viz_type='umap', N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_kmeanspp(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        #gene_names=viz_genes, genes=genes,
        #gene_expr=vstack(datasets),
        viz_type='umap', N_only=20000, kmeans=False, visualize_orig=False
    )
    
    from geosketch import gs, uniform, srs
    samp_idx = gs(X_dimred, 20000, replace=False)
    #samp_idx = uniform(X_dimred, 20000, replace=False)
    #samp_idx = srs(X_dimred, 20000, replace=False)
    
    viz_genes = [
        'SLC1A3',# 'GJA1', 'MBP', 'PLP1', 'TRF',
        #'GJA1', 'MBP', 'PLP1', 'TRF',
        #'CST3', 'CPE', 'FTH1', 'APOE', 'MT1', 'NDRG2', 'TSPAN7',
        #'PLP1', 'MAL', 'PTGDS', 'CLDN11', 'APOD', 'QDPR', 'MAG', 'ERMN',
        #'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        #'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        #'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    X = np.log1p(normalize(X[samp_idx, :]))

    embedding = visualize(
        [ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
        NAMESPACE + '_astro{}'.format(len(samp_idx)),
        [ str(ct) for ct in sorted(set(cell_labels)) ],
        gene_names=viz_genes, gene_expr=X, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png',
        #viz_cluster=True
    )
    
    cell_labels = (
        open('data/cell_labels/zeisel_louvain.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    
