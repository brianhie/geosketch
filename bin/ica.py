import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

np.random.seed(0)

NAMESPACE = 'ica'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/ica/ica_bone_marrow_h5',
    'data/ica/ica_cord_blood_h5',
]

if __name__ == '__main__':
    save_sketch(data_names, NAMESPACE)
    exit()
    
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    if not os.path.isfile('data/dimred_{}.txt'.format(NAMESPACE)):
        log('Scanorama integration...')
        datasets_dimred, genes = process_data(datasets, genes)
        datasets_dimred = assemble(datasets_dimred, knn=100, sigma=50,
                                   batch_size=25000)
        X_dimred = np.concatenate(datasets_dimred)
        np.savetxt('data/dimred_{}.txt'.format(NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred_{}.txt'.format(NAMESPACE))

    viz_genes = [
        'CD14', 'PTPRC', 'FCGR3A', 'ITGAX', 'ITGAM', 'CD19', 'HLA-DRB1',
        'FCGR2B', 'FCGR2A', 'CD3E', 'CD4', 'CD8A','CD8B', 'CD28', 'CD8',
        'TBX21', 'IKAROS', 'IL2RA', 'CD44', 'SELL', 'CCR7', 'MS4A1',
        'CD68', 'CD163', 'IL5RA', 'SIGLEC8', 'KLRD1', 'NCR1', 'CD22',
        'IL3RA', 'CCR6', 'IL7R', 'CD27', 'FOXP3', 'PTCRA', 'ID3', 'PF4',
        'CCR10', 'SIGLEC7', 'NKG7', 'S100A8', 'CXCR3', 'CCR5', 'CCR3',
        'CCR4', 'PTGDR2', 'RORC'
    ]

    experiment_gs(X_dimred, NAMESPACE, perplexity=1000,
                  gene_names=viz_genes, genes=genes,
                  gene_expr=vstack(datasets),
                  visualize_orig=False,
                  kmeans=False)

    log('Done.')
