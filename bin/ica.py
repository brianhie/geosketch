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
    if not os.path.isfile('data/dimred_{}.txt'.format(NAMESPACE)):
        datasets, genes_list, n_cells = load_names(data_names)
        datasets, genes = merge_datasets(datasets, genes_list)
        log('Scanorama integration...')
        datasets_dimred, genes = process_data(datasets, genes)
        datasets_dimred = assemble(datasets_dimred, knn=100, sigma=50,
                                   batch_size=25000)
        X_dimred = np.concatenate(datasets_dimred)
        np.savetxt('data/dimred_{}.txt'.format(NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred_{}.txt'.format(NAMESPACE))

    experiment_srs(X_dimred, NAMESPACE, perplexity=1000,
                   visualize_orig=False, kmeans=False)
    
    log('Done.')
