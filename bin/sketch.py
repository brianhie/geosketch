from annoy import AnnoyIndex
from intervaltree import Interval, IntervalTree
import numpy as np
from sklearn.preprocessing import normalize
import sys

def gs(X, N, seed=None, replace=False, prenormalized=False):
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
            query = (np.random.normal(size=(n_features))
                     .reshape(1, -1).astype('float32'))
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
