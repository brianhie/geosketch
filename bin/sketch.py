from annoy import AnnoyIndex
from intervaltree import Interval, IntervalTree
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import sys

from utils import log

def gs(X, N, seed=None, replace=True, method='faiss', verbose=False, labels=None):
    k = 10
    power = 1

    if verbose:
        log('k = {}, power = {}'.format(k, power))
    
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    method = check_method(method)
    if method == 'faiss':
        import faiss
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)
    X = np.ascontiguousarray(X, dtype='float32')

    index = AnnoyIndex(n_features, metric='manhattan')
    for i in range(n_samples):
        index.add_item(i, X[i, :])
    index.build(10)

    #quantizer = faiss.IndexFlatL2(n_features)
    #index = faiss.IndexIVFFlat(quantizer, n_features, 100, faiss.METRIC_L2)
    #index.train(X)
    #index.add(X)

    weights = []
    for i in range(n_samples):
        if verbose and i % 20000 == 0:
            log('Process {} samples'.format(i))
        #dist, idx = index.search(X[i, :].reshape(1, -1), k)
        idx, dist = index.get_nns_by_item(i, k, include_distances=True)
        weights.append(np.min(dist[1]) ** power)
    weights = np.array(weights) / sum(weights)

    if labels is not None:
        clusters = sorted(set(labels))
        for cluster in clusters:
            print('Cluster {} has mean {} and std {}'
                  .format(cluster, np.mean(weights[labels == cluster]),
                          np.std(weights[labels == cluster])))
            
    return np.random.choice(range(n_samples), size=N, p=weights, replace=replace)

def gs_binary(X, N, seed=None, replace=False, prenormalized=False, method='faiss',
              alpha=0.1, max_iter=200, verbose=3, labels=None):
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    method = check_method(method)
    if method == 'faiss':
        import faiss
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)
    X = np.ascontiguousarray(X, dtype='float32')

    rand100 = np.random.choice(n_samples, size=100)
    dist100 = pairwise_distances(X[rand100, :])
    radius = np.mean(dist100[dist100 != 0])

    if verbose > 1:
        log('Initializing radius to {}'.format(radius))

    quantizer = faiss.IndexFlatL2(n_features)
    index = faiss.IndexIVFFlat(quantizer, n_features, 100, faiss.METRIC_L2)
    index.train(X)
    
    all_samples = set(range(n_samples))
    filtered_samples = {}
    centers = []

    low_radius, high_radius = 0., None
    
    n_iter = 0
    while True:

        if verbose > 1:
            log('n_iter = {}'.format(n_iter))

        permuted_samples = np.random.permutation(
            list(all_samples - set(filtered_samples.keys()))
        )
        
        for sample_idx, sample in enumerate(permuted_samples):
            if verbose > 1:
                if sample_idx % 10000 == 0:
                    log('sample_idx = {}'.format(sample_idx))
            
            x_sample = X[sample, :].reshape(1, -1)
            if len(filtered_samples) > 1:
                dist, idx = index.search(x_sample, 1)
                if dist[0][0] > radius:
                    # Use sample as new cluster center in data structure.
                    index.add(x_sample)
                    filtered_samples[sample] = [ sample ]
                    centers.append(sample)
                else:
                    # Add sample to cluster.
                    filtered_samples[centers[idx[0][0]]].append(sample)
            else:
                # Initialize data structure.
                index.add(x_sample)
                filtered_samples[sample] = [ sample ]
                centers.append(sample)

        if verbose:
            log('Found {} cluster centers'.format(len(filtered_samples)))

        if len(filtered_samples) > N * (1 + alpha):
            # Too many cluster centers, increase cluster radius and try again.
            if verbose:
                log('More clusters than {}, increase radius to {}'
                    .format(N * (1 + alpha), radius * 2. if high_radius is None else
                            (high_radius + radius) / 2.))

            low_radius = radius
            if high_radius is None:
                radius *= 2.
            else:
                radius = (high_radius + radius) / 2.
                
            index = faiss.IndexHNSWFlat(n_features, 32)
            #all_samples = set(filtered_samples.keys())
            filtered_samples = {}
            centers = []

        elif len(filtered_samples) < N / (1 + alpha):
            # Too few cluster centers, decrease cluster radius and add more points.
            if verbose:
                log('Fewer clusters than {}, decrease radius to {}'
                    .format(N / (1 + alpha), (radius + low_radius) / 2.))

            high_radius = radius
            radius = (radius + low_radius) / 2.

        else:
            break

        if n_iter > max_iter:
            # Should rarely get here.
            sys.stderr.write('WARNING: Max iterations reached, try increasing '
                             ' alpha parameter.\n')
            break
        n_iter += 1

    log('Found {} cluster centers'.format(len(filtered_samples)))

    clusters = sorted(set(labels))
    
    for center in filtered_samples:
        samples = labels[filtered_samples[center]]
        for cluster in clusters:
            print('cluster has {} in center {}'
                  .format(sum(samples == cluster), center))
        print('')
    exit()
        
    # Sample `N' cluster centers.
    centers = list(filtered_samples.keys())
    weights = np.array([ 1. / len(filtered_samples[center]) for center in centers ])
    N_centers = np.random.choice(centers, size=N, p=(weights / np.sum(weights)))
    
#    if len(filtered_samples) < N:
#        center_idx = range(len(centers))
#        center_idx += list(np.random.choice(
#            range(len(centers)), size=(N - len(filtered_samples))
#        ))
#    else:
#        center_idx = np.random.choice(range(len(centers)), size=N, replace=True)

    
    # For `N' cluster centers, sample a point near that center.
    gs_idx = []
    for center in N_centers:
        samples = filtered_samples[center]
        gs_idx.append(np.random.choice(samples, size=1)[0])

    return sorted(gs_idx)
    
def gs1(X, N, seed=None, replace=False, prenormalized=False):
    try:
        import faiss
    except ImportError:
        sys.stderr.write(
            'ERROR: Please install faiss: '
            'https://github.com/facebookresearch/faiss/blob/master/INSTALL.md\n'
        )
        exit(1)

    n_samples, n_features = X.shape

    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)

    if not seed is None:
        np.random.seed(seed)

    #if not prenormalized:
    #    X = normalize(X, norm='l2', axis=1)
    X = np.ascontiguousarray(X, dtype='float32')

    # Build index.
    quantizer = faiss.IndexFlatL2(n_features)
    index = faiss.IndexIVFFlat(quantizer, n_features, 100,
                               faiss.METRIC_L2)
    index.train(X)
    index.add(X)

    # Generate Gaussian noise and use it to query data structure.
    gs_idx = []
    n_retries = N
    for i in range(N):
        for j in range(n_retries):
            query = np.random.normal(size=(n_features))
            #query = query / np.linalg.norm(query)
            query = query.reshape(1, -1).astype('float32')
            _, I = index.search(query, 1)
            assert(len(I) == 1)
            assert(len(I[0]) == 1)
            k_argmax = I[0][0]
            if k_argmax != -1:
                break
        assert(k_argmax != -1)

        if not replace:
            n_removed = index.remove_ids(
                faiss.IDSelectorRange(k_argmax, k_argmax + 1)
            )
            assert(n_removed == 1)
        
        gs_idx.append(k_argmax)

    if not replace:
        assert(len(set(gs_idx)) == N)

    return gs_idx

def gs2(X, N, seed=None, replace=True, n_retries=None,
        method='annoy', n_sites=None):

    from kmeanspp import kmeanspp
    
    return kmeanspp(X, N)
    
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    method = check_method(method)
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)
    if n_retries is None:
        n_retries = max(int(np.log2(n_samples)) * 2, 100)
    if n_sites is None:
        n_sites = min(int(X.shape[0] / 10), 100)

    # Feature bounds.
    X = np.ascontiguousarray(X, dtype='float32')
    X_max = X.max(0)
    X_min = X.min(0)

    import faiss

    n_outline = 100
    outline_index = faiss.IndexHNSWFlat(n_features, 32)
    outline_points = []
    for i in range(n_outline):
        outline_points.append(np.random.uniform(low=X_min, high=X_max))
    outline_index.add(np.array(outline_points).astype('float32'))
        
    covered_points = set()
    for i in range(X.shape[0]):
        x = X[i, :].reshape(1, -1).astype('float32')
        outline_point = outline_index.search(x, 1)[1][0][0]
        covered_points.add(outline_point)
    print('{} out of {} points are covered'
          .format(len(covered_points), n_outline))

    index = faiss.IndexHNSWFlat(n_features, 32)
    index.add(X)
    
    if replace:
        n_iter = N
        gs_idx = []
    else:
        n_iter = int(1e5)
        gs_idx = set()
        
    for i in range(n_iter):

        while True:
            query = np.random.uniform(low=X_min, high=X_max)
            query = query.astype('float32').reshape(1, -1)
            outline_point = outline_index.search(query, 1)[1][0][0]
            if outline_point in covered_points:
                break
        
        idx = index.search(query, 1)[1][0][0]

        if replace:
            gs_idx.append(idx)
        else:
            gs_idx.add(idx)
            if len(gs_idx) >= N:
                break

    return sorted(gs_idx)

def check_method(method):
    if method == 'faiss':
        try:
            import faiss
        except ImportError:
            sys.stderr.write(
                'WARNING: Consider installing faiss for faster sampling: '
                'https://github.com/facebookresearch/faiss/blob/master/INSTALL.md\n'
            )
            sys.stderr.write('Defaulting to annoy.\n')
            method = 'annoy'
    elif method != 'annoy':
        method = 'annoy'
    return method

def index_annoy(X):
    index = AnnoyIndex(X.shape[1], metric='euclidean')
    for i in range(X.shape[0]):
        index.add_item(i, X[i, :])
    index.build(10)
    return index
            
def index_faiss(X):
    import faiss
    X = np.ascontiguousarray(X, dtype='float32')
    quantizer = faiss.IndexFlatL2(X.shape[1])
    index = faiss.IndexIVFFlat(quantizer, X.shape[1], 100,
                               faiss.METRIC_L2)
    index.train(X)
    index.add(X)
    return index

def query_faiss(query, n_query, index):
    query = query.reshape(1, -1)
    query = np.ascontiguousarray(query, dtype='float32')
    near_dist, near_idx = index.search(query, n_query)
    assert(len(near_dist) == 1)
    assert(len(near_idx) == 1)
    if -1 in near_idx[0]:
        return [], []
    if len(near_idx[0]) > len(set(near_idx[0])):
        index.nprobe *= 2
    return near_idx[0], near_dist[0]

def query_annoy(query, n_query, index):
    return index.get_nns_by_vector(
        query, n_query, include_distances=True
    )

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
