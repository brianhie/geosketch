import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import sys

from .kmeanspp import kmeanspp
from .utils import log

def gs(X, N, **kwargs):
    return gs_gap(X, N, **kwargs)

def gs_gap(X, N, k='auto', seed=None, replace=False,
           alpha=0.1, max_iter=200, verbose=0, labels=None):
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)
    if k == 'auto':
        k = int(np.sqrt(n_samples))

    X = X - X.min(0)
    X /= X.max()

    X_ptp = X.ptp(0)

    low_unit, high_unit = 0., max(X_ptp)
    
    unit = (low_unit + high_unit) / 4.

    n_iter = 0
    while True:

        if verbose > 1:
            log('n_iter = {}'.format(n_iter))

        grid_table = np.zeros((n_samples, n_features))

        for d in range(n_features):
            if X_ptp[d] <= unit:
                continue
                
            points_d = X[:, d]
            curr_start = None
            curr_interval = -1
            for sample_idx in np.argsort(points_d):
                if curr_start is None or \
                   curr_start + unit < points_d[sample_idx]:
                    curr_start = points_d[sample_idx]
                    curr_interval += 1
                grid_table[sample_idx, d] = curr_interval

        grid = {}

        for sample_idx in range(n_samples):
            grid_cell = tuple(grid_table[sample_idx, :])
            if grid_cell not in grid:
                grid[grid_cell] = []
            grid[grid_cell].append(sample_idx)

        del grid_table

        if verbose:
            log('Found {} non-empty grid cells'.format(len(grid)))
            
        if len(grid) > k * (1 + alpha):
            # Too many grid cells, increase unit.
            low_unit = unit
            if high_unit is None:
                unit *= 2.
            else:
                unit = (unit + high_unit) / 2.
            
            if verbose:
                log('Grid size {}, increase unit to {}'
                    .format(len(grid), unit))

        elif len(grid) < k / (1 + alpha):
            # Too few grid cells, decrease unit.
            high_unit = unit
            if low_unit is None:
                unit /= 2.
            else:
                unit = (unit + low_unit) / 2.
                
            if verbose:
                log('Grid size {}, decrease unit to {}'
                    .format(len(grid), unit))
        else:
            break

        if high_unit is not None and low_unit is not None and \
           high_unit - low_unit < 1e-20:
            break
        
        if n_iter >= max_iter:
            # Should rarely get here.
            sys.stderr.write('WARNING: Max iterations reached, try increasing '
                             ' alpha parameter.\n')
            break
        n_iter += 1

    if verbose:
        log('Found {} grid cells'.format(len(grid)))

    gs_idx = []
    for n in range(N):
        grid_cells = list(grid.keys())
        grid_cell = grid_cells[np.random.choice(len(grid_cells))]
        samples = list(grid[grid_cell])
        sample = samples[np.random.choice(len(samples))]
        if not replace:
            grid[grid_cell].remove(sample)
            if len(grid[grid_cell]) == 0:
                del grid[grid_cell]
        gs_idx.append(sample)

    return sorted(gs_idx)

def gs_grid(X, N, k='auto', seed=None, replace=False,
            alpha=0.1, max_iter=200, verbose=0, labels=None):
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)
    if k == 'auto':
        k = int(np.sqrt(n_samples))

    X = X - X.min(0)
    X /= X.max()

    low_unit, high_unit = 0., np.max(X)
    
    unit = (low_unit + high_unit) / 4.

    n_iter = 0
    while True:

        if verbose > 1:
            log('n_iter = {}'.format(n_iter))

        grid = {}

        unit_d = unit# * n_features

        for sample_idx in range(n_samples):
            if verbose > 1:
                if sample_idx % 10000 == 0:
                    log('sample_idx = {}'.format(sample_idx))
            
            sample = X[sample_idx, :]

            grid_cell = tuple(np.floor(sample / unit_d).astype(int))

            if grid_cell not in grid:
                grid[grid_cell] = set()
            grid[grid_cell].add(sample_idx)

        if verbose:
            log('Found {} non-empty grid cells'.format(len(grid)))
            
        if len(grid) > k * (1 + alpha):
            # Too many grid cells, increase unit.
            low_unit = unit
            if high_unit is None:
                unit *= 2.
            else:
                unit = (unit + high_unit) / 2.
            
            if verbose:
                log('Grid size {}, increase unit to {}'
                    .format(len(grid), unit))

        elif len(grid) < k / (1 + alpha):
            # Too few grid cells, decrease unit.
            high_unit = unit
            if low_unit is None:
                unit /= 2.
            else:
                unit = (unit + low_unit) / 2.
                
            if verbose:
                log('Grid size {}, decrease unit to {}'
                    .format(len(grid), unit))
        else:
            break

        if high_unit is not None and low_unit is not None and \
           high_unit - low_unit < 1e-20:
            break
        
        if n_iter >= max_iter:
            # Should rarely get here.
            sys.stderr.write('WARNING: Max iterations reached, try increasing '
                             ' alpha parameter.\n')
            break
        n_iter += 1

    if verbose:
        log('Found {} grid cells'.format(len(grid)))
                
    gs_idx = []
    for n in range(N):
        grid_cells = list(grid.keys())
        grid_cell = grid_cells[np.random.choice(len(grid_cells))]
        samples = list(grid[grid_cell])
        sample = samples[np.random.choice(len(samples))]
        if not replace:
            grid[grid_cell].remove(sample)
            if len(grid[grid_cell]) == 0:
                del grid[grid_cell]
        gs_idx.append(sample)

    return sorted(gs_idx)

def gs_exact(X, N, k='auto', seed=None, replace=False,
             tol=1e-3, n_iter=300, verbose=1):
    ge_idx = gs(X, N, replace=replace)
    
    dist = pairwise_distances(X, n_jobs=-1)
    
    cost = dist.max()

    iter_i = 0
    
    while iter_i < n_iter:

        if verbose:
            log('iter_i = {}'.format(iter_i))

        labels = np.argmin(dist[ge_idx, :], axis=0)

        ge_idx_new = []
        for cluster in range(N):
            cluster_idx = np.nonzero(labels == cluster)[0]
            if len(cluster_idx) == 0:
                ge_idx_new.append(ge_idx[cluster])
                continue
            X_cluster = dist[cluster_idx, :]
            X_cluster = X_cluster[:, cluster_idx]
            within_idx = np.argmin(X_cluster.max(0))
            ge_idx_new.append(cluster_idx[within_idx])
        ge_idx = ge_idx_new

        cost, prev_cost = dist[ge_idx, :].min(0).max(), cost
        assert(cost <= prev_cost)

        if prev_cost - cost < tol:
            break

        iter_i += 1

    return ge_idx

def srs_center(X, N, **kwargs):
    return srs(X - X.mean(0), N, **kwargs)

def srs_positive(X, N, **kwargs):
    return srs(X - X.min(0), N, **kwargs)

def srs_unit(X, N, **kwargs):
    X = X - X.min(0)
    X /= X.max()
    return srs(X, N, **kwargs)

def srs(X, N, seed=None, replace=False, prenormalized=False):
    n_samples, n_features = X.shape

    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)

    if not seed is None:
        np.random.seed(seed)

    if not prenormalized:
        X = normalize(X).astype('float32')

    srs_idx = []
    for i in range(N):
        Phi_i = np.random.normal(size=(n_features))
        Phi_i /= np.linalg.norm(Phi_i)
        Q_i = X.dot(Phi_i)
        if not replace:
            Q_i[srs_idx] = 0
        k_argmax = np.argmax(np.absolute(Q_i))
        srs_idx.append(k_argmax)

    return srs_idx

def uniform(X, N, seed=None, replace=False):
    n_samples, n_features = X.shape

    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)

    if not seed is None:
        np.random.seed(seed)
        
    return list(np.random.choice(n_samples, size=N, replace=replace))

def kmeans(X, N, seed=None, replace=False, init='random'):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(np.sqrt(X.shape[0])), init=init,
                n_init=1, random_state=seed)
    km.fit(X)

    louv = {}
    for i, cluster in enumerate(km.labels_):
        if cluster not in louv:
            louv[cluster] = []
        louv[cluster].append(i)
    
    lv_idx = []
    for n in range(N):
        louv_cells = list(louv.keys())
        louv_cell = louv_cells[np.random.choice(len(louv_cells))]
        samples = list(louv[louv_cell])
        sample = samples[np.random.choice(len(samples))]
        if not replace:
            louv[louv_cell].remove(sample)
            if len(louv[louv_cell]) == 0:
                del louv[louv_cell]
        lv_idx.append(sample)

    return lv_idx

def kmeansppp(X, N, seed=None, replace=False):
    return kmeans(X, N, seed=seed, replace=replace, init='k-means++')

def louvain1(X, N, seed=None, replace=False):
    return louvain(X, N, resolution=1, seed=seed, replace=replace)

def louvain3(X, N, seed=None, replace=False):
    return louvain(X, N, resolution=3, seed=seed, replace=replace)
    
def louvain(X, N, resolution=1, seed=None, replace=False):
    from anndata import AnnData
    import scanpy.api as sc
    
    adata = AnnData(X=X)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.louvain(adata, resolution=resolution, key_added='louvain')
    cluster_labels_full = adata.obs['louvain'].tolist()

    louv = {}
    for i, cluster in enumerate(cluster_labels_full):
        if cluster not in louv:
            louv[cluster] = []
        louv[cluster].append(i)
    
    lv_idx = []
    for n in range(N):
        louv_cells = list(louv.keys())
        louv_cell = louv_cells[np.random.choice(len(louv_cells))]
        samples = list(louv[louv_cell])
        sample = samples[np.random.choice(len(samples))]
        if not replace:
            louv[louv_cell].remove(sample)
            if len(louv[louv_cell]) == 0:
                del louv[louv_cell]
        lv_idx.append(sample)

    return lv_idx
    
def label(X, sites, site_labels, approx=True):
    if approx:
        return label_approx(X, sites, site_labels)
    else:
        return label_exact(X, sites, site_labels)

def label_exact(X, sites, site_labels):
    assert(sites.shape[0] > 0)
    assert(X.shape[1] == sites.shape[1])

    labels = []
    for i in range(X.shape[0]):
        nearest_site = None
        min_dist = None
        for j in range(sites.shape[0]):
            dist = np.sum((X[i, :] - sites[j, :])**2)
            if min_dist is None or dist < min_dist:
                nearest_site = j
                min_dist = dist
        assert(not nearest_site is None)
        labels.append(site_labels[nearest_site])
    return np.array(labels)

def label_approx(X, sites, site_labels):
    from annoy import AnnoyIndex
    
    assert(X.shape[1] == sites.shape[1])

    # Build index over site points.
    aindex = AnnoyIndex(sites.shape[1], metric='euclidean')
    for i in range(sites.shape[0]):
        aindex.add_item(i, sites[i, :])
    aindex.build(10)

    labels = []
    for i in range(X.shape[0]):
        # Find nearest site point.
        nearest_site = aindex.get_nns_by_vector(X[i, :], 1)
        if len(nearest_site) < 1:
            labels.append(None)
            continue
        labels.append(site_labels[nearest_site[0]])
        
    return np.array(labels)
