from anndata import AnnData
import numpy as np
from scanorama import *
import scanpy.api as sc
from sklearn.metrics import silhouette_samples as sil
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'shahin'

data_names = [
    'data/exported_T_cells/sample1',
    'data/exported_T_cells/sample2',
    'data/exported_T_cells/sample3',
    'data/exported_T_cells/sample4',
    'data/exported_T_cells/sample5',
]

if __name__ == '__main__':
    from process import process
    process(data_names, min_trans=0)
    
    datasets, genes_list, n_cells = load_names(data_names)
    
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )

    X_dimred = np.concatenate(datasets_dimred)
    
    cell_labels = (
        open('data/cell_labels/shahin_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_
    
    adata = AnnData(X=X_dimred)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    embedding = np.array(adata.obsm['X_umap'])
            
    visualize(None,
              labels, NAMESPACE + '_ds', names,
              image_suffix='.png',
              embedding=embedding)
    
    visualize(None,
              cell_labels, NAMESPACE + '_type', cell_types,
              image_suffix='.png',
              embedding=embedding)

    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)
    
    X_dimred = np.concatenate(datasets_dimred)

    adata = AnnData(X=X_dimred)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    embedding = np.array(adata.obsm['X_umap'])
            
    visualize(None,
              labels, NAMESPACE + '_ds_uncorrected', names,
              image_suffix='.png',
              embedding=embedding)
    
    visualize(None,
              cell_labels, NAMESPACE + '_type_uncorrected', cell_types,
              image_suffix='.png',
              embedding=embedding)
