import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'pbmc'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/10x/68k_pbmc',
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/pbmc_10X',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    
    log('Scanorama integration...')
    datasets_dimred, genes = process_data(datasets, genes)
    datasets_dimred = assemble(datasets_dimred)
    X_dimred = np.concatenate(datasets_dimred)

    viz_genes = [
        'CD14', 'PTPRC', 'FCGR3A', 'ITGAX', 'ITGAM', 'CD19', 'HLA-DRB1',
        'FCGR2B', 'FCGR2A', 'CD3E', 'CD4', 'CD8A','CD8B', 'CD28', 'CD8',
        'TBX21', 'IKAROS', 'IL2RA', 'CD44', 'SELL', 'CCR7', 'MS4A1',
        'CD68', 'CD163', 'IL5RA', 'SIGLEC8', 'KLRD1', 'NCR1', 'CD22',
        'IL3RA', 'CCR6', 'IL7R', 'CD27', 'FOXP3', 'PTCRA', 'ID3', 'PF4',
        'CCR10', 'SIGLEC7', 'NKG7', 'S100A8', 'CXCR3', 'CCR5', 'CCR3',
        'CCR4', 'PTGDR2', 'RORC'
    ]

    experiment_gs(X_dimred, NAMESPACE, perplexity=100,
                  gene_names=viz_genes, genes=genes,
                  gene_expr=vstack(datasets),
                  visualize_orig=False,
                  kmeans=False)

    log('Done.')
