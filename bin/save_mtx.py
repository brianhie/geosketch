import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from utils import mkdir_p

def save_mtx(dir_name, X, genes):
    X = X.tocoo()
    
    if not os.path.exists(dir_name):
        mkdir_p(dir_name)

    with open(dir_name + '/matrix.mtx', 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        
        f.write('{} {} {}\n'.format(X.shape[1], X.shape[0], X.nnz))

        try:
            from itertools import izip
        except ImportError:
            izip = zip
        
        for i, j, val in izip(X.row, X.col, X.data):
            f.write('{} {} {}\n'.format(j + 1, i + 1, int(val)))

    with open(dir_name + '/genes.tsv', 'w') as f:
        for idx, gene in enumerate(genes):
            f.write('{}\t{}\n'.format(idx + 1, gene))

    with open(dir_name + '/barcodes.tsv', 'w') as f:
        for idx in range(X.shape[0]):
            f.write('cell{}-1\n'.format(idx))
            
if __name__ == '__main__':
    from mouse_brain import data_names

    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)

    save_mtx('data/mouse_brain', vstack(datasets), genes)
