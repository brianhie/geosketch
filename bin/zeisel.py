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
    from process import process
    process(data_names, min_trans=0)
    
    datasets, genes_list, n_cells = load_names(data_names)
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

    viz_genes = [
        'GJA1', 'MBP',
        'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    cell_labels = (
        open('data/cell_labels/zeisel_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_names = sorted(set(cell_labels))
    cell_labels = le.transform(cell_labels)
    
    from ample import gs_gap
    samp_idx = gs_gap(X_dimred, 20000, replace=False)
    
    X_samp = normalize(X[samp_idx, :])

    embedding = visualize(
        [ X_dimred[samp_idx, :] ], cell_labels,
        NAMESPACE + '_astro{}'.format(len(samp_idx)), cell_names,
        gene_names=viz_genes, gene_expr=X_samp, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png'
    )
