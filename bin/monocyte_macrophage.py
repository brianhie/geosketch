import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'mono_macro'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/macrophage/monocytes_1',
    'data/macrophage/monocytes_2',
    'data/macrophage/monocytes_3',
    'data/macrophage/monocytes_4',
    'data/pbmc/10x/cd14_monocytes',
    'data/macrophage/mcsf_day3_1',
    'data/macrophage/mcsf_day3_2',
    'data/macrophage/mcsf_day6_1',
    'data/macrophage/mcsf_day6_2',
    'data/macrophage/mcsf_day6_3',
    'data/macrophage/mcsf_day6_4',
]

def write_table(dataset, genes, name):
    prefix = name.split('/')[-1]
    with open(name + '_table.txt', 'w') as f:
        header = '\t'.join([ prefix + str(i) for i in range(dataset.shape[0]) ])
        f.write(header + '\n')

        for i in range(dataset.shape[1]):
            line = '\t'.join([ str(int(j)) for j in dataset[:, i] ])
            f.write(genes[i] + '\t' + line + '\n')
            
def keep_valid():
    with open('data/pseudotime/GSE72857_umitab.txt') as f:
        all_cells = f.readline().rstrip().split()[1:]

    with open('data/pseudotime/meta.txt') as f:
        cell_to_type = []
        for line in f:
            fields = line.rstrip().split()
            cell_to_type[fields[0]] = fields[1]

    valid_idx = []
    cell_names = []
    for c_idx, cell in enumerate(all_cells):
        if cell in cell_to_type:
            valid_idx.append(c_idx)
            cell_names.append(cell_to_type[cell])
            
    return valid_idx, cell_names
    
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
    
    cell_names = []
    for i, a in enumerate(datasets):
        if 'monocyte' in data_names[i]:
            cell_names += [ 'monocyte' ] * a.shape[0]
        elif 'day3' in data_names[i]:
            cell_names += [ 'mcsf_day3' ] * a.shape[0]
        elif 'day6' in data_names[i]:
            cell_names += [ 'mcsf_day6' ] * a.shape[0]
        else:
            assert(False)
    le = LabelEncoder().fit(cell_names)
    cell_labels = le.transform(cell_names)
    
    write_table(X.toarray(), genes, 'data/pseudotime/' + NAMESPACE)

    with open('data/pseudotime/mono_macro_meta.txt', 'w') as of:
        of.write('Label\n')
        for idx in range(X.shape[0]):
            of.write('mono_macro{}\t{}'.format(idx, cell_names[idx]))
            
    from geosketch import gs, gs_gap, uniform

    gs_idx = gs(X_dimred, 110, replace=False)
    write_table(X[gs_idx, :].toarray(), genes, 'data/pseudotime/' + NAMESPACE + '_gs')
    report_cluster_counts(cell_labels[gs_idx])

    with open('data/pseudotime/mono_macro_meta_gs.txt', 'w') as of:
        of.write('Label\n')
        i = 0
        for idx in range(X.shape[0]):
            if idx not in gs_idx:
                continue
            of.write('mono_macro_gs{}\t{}\n'.format(i, cell_names[idx]))
            i += 1
    
    uni_idx = uniform(X_dimred, 110, replace=False)
    write_table(X[uni_idx, :].toarray(), genes, 'data/pseudotime/' + NAMESPACE + '_uni')
    report_cluster_counts(cell_labels[uni_idx])
    
    with open('data/pseudotime/mono_macro_meta_uni.txt', 'w') as of:
        of.write('Label\n')
        i = 0
        for idx in range(X.shape[0]):
            if idx not in uni_idx:
                continue
            of.write('mono_macro_uni{}\t{}\n'.format(i, cell_names[idx]))
            i += 1

    with open('data/pseudotime/mono_macro_genes.txt', 'w') as of:
        of.write('gene_short_name\n')
        for gene in genes:
            of.write('{}\t{}\n'.format(gene, gene))
