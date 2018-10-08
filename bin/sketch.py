from annoy import AnnoyIndex
import numpy as np
import sys

def srs(X, N, seed=None):
    if not seed is None:
        np.random.seed(seed)
    
    n_samples, n_features = X.shape

    srs_idx = []
    for i in range(N):
        Phi_i = np.random.normal(size=(n_features))
        Q_i = X.dot(Phi_i)
        Q_i[srs_idx] = 0
        k_argmax = np.argmax(np.absolute(Q_i))
        srs_idx.append(k_argmax)

    return srs_idx

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
