import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def differential_entropies(X, labels):
    n_samples, n_features = X.shape
    
    labels = np.array(labels)
    names = sorted(set(labels))

    entropies = []
    
    for name in names:
        name_idx = np.where(labels == name)[0]

        gm = GaussianMixture().fit(X[name_idx, :])

        mn = multivariate_normal(
            mean=gm.means_.flatten(),
            cov=gm.covariances_.reshape(n_features, n_features)
        )

        entropies.append(mn.entropy())

    probs = softmax(entropies)

    for name, entropy, prob in zip(names, entropies, probs):
        #print('{}\t{}\t{}'.format(name, entropy, prob))
        print('{}\t{}'.format(name, entropy))
