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

def centroid_label(X, centroids, centroid_labels):
    assert(X.shape[1] == centroids.shape[1])

    labels = []
    for i in range(X.shape[0]):
        min_centroid = None
        min_dist = None
        for j in range(centroids.shape[0]):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if min_dist is None or dist < min_dist:
                min_centroid = j
                dist = min_dist
            labels.append(centroid_labels[min_centroid])
    return np.array(labels)
