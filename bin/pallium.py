import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'pallium'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/murine_atlases/neuron_1M/neuron_1M',
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

    from geosketch import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '1000')
    
    for scale in [ 0.05, 0.025, 0.01 ]:
        N = int(X.shape[0] * scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + str(N))

    exit()
        
    viz_genes = [
        'GJA1', 'MBP',
        'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    from geosketch import gs_gap
    samp_idx = gs_gap(X_dimred, 20000, replace=False)
    
    X_samp = normalize(X[samp_idx, :])

    embedding = visualize(
        [ X_dimred[samp_idx, :] ], np.zeros(len(samp_idx), dtype=int),
        NAMESPACE + '_astro{}'.format(len(samp_idx)), [ '0' ],
        gene_names=viz_genes, gene_expr=X_samp, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png'
    )
    
    from geosketch import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '1000')
    
    for scale in [ 0.05, 0.025, 0.01 ]:
        N = int(X.shape[0] * scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + str(N))

    
