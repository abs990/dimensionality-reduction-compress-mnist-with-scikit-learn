"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist_subset
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

def load_mnist_subset(n=3000):
    # cache mnist.npz in tempfile.gettempdir(); return (X (n, 784) float64, y int64).
    path = Path(tempfile.gettempdir()) / "mnist.npz"

    if not path.exists():
        urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)

    with np.load(path) as data:
        x_train = data["x_train"][:n].reshape(n, -1)
        y_train = data["y_train"][:n]

    return x_train.astype(np.float64), y_train.astype(np.int64)

# Step 2 - fit_pca
from sklearn.decomposition import PCA

def fit_pca(X, n_components=None, random_state=42):
    return PCA(n_components=n_components, random_state=random_state).fit(X)

# Step 3 - components_for_variance
def components_for_variance(pca, threshold=0.95):
    # first index where the cumulative explained variance reaches threshold, plus one.
    return 1 + int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold))

# Step 4 - explained_variance_curve
def explained_variance_curve(pca, ks):
    # {k: cumulative explained variance of the first k components, rounded to 4 decimals}.
    c_sum = np.cumsum(pca.explained_variance_ratio_)
    return {
        k: round(float(c_sum[k - 1]), 4) for k in ks
    }

# Step 5 - compress
def compress(X, n_components, random_state=42):
    # fit PCA with n_components; return (pca, pca.transform(X)).
    pca = fit_pca(X, n_components, random_state)
    return pca, pca.transform(X)

# Step 6 - reconstruction_error
def reconstruction_error(pca, X):
    # mean squared per-pixel error between X and its PCA reconstruction.
    return float(np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2))

# Step 7 - compression_ratio
def compression_ratio(X, X_reduced):
    # original width / reduced width, rounded to 2 decimals.
    return round(float(X.shape[1] / X_reduced.shape[1]), 2)

# Step 8 - randomized_pca
def randomized_pca(X, n_components, random_state=42):
    # PCA(svd_solver='randomized'); return {'pca': ..., 'variance_gap': |sum ratio - exact sum ratio|}.
    compress_pca, _ = compress(X, n_components, random_state)
    pca = PCA(n_components, svd_solver="randomized", random_state=random_state).fit(X)
    return {
        "pca": pca,
        "variance_gap": float(np.abs(pca.explained_variance_ratio_.sum() - compress_pca.explained_variance_ratio_.sum()))
    }

# Step 9 - incremental_pca
from sklearn.decomposition import IncrementalPCA

def incremental_pca(X, n_components, n_batches=10):
    # IncrementalPCA fitted batch by batch with partial_fit.
    ipca = IncrementalPCA(n_components=n_components)
    batches = np.array_split(X, n_batches)
    for batch in batches:
        ipca.partial_fit(batch)
    return ipca

# Step 10 - random_projection
from sklearn.random_projection import GaussianRandomProjection

def random_projection(X, n_components, random_state=42, n_pairs=500):
    # GaussianRandomProjection; distance-ratio statistics on random pairs.
    result = {}

    grp = GaussianRandomProjection(n_components=n_components, random_state=random_state).fit(X)
    Xr = grp.transform(X)

    rng = np.random.default_rng(random_state)
    indices = np.array(rng.integers(0, len(X), (n_pairs, 2)))
    mask = (indices[:, 0] != indices[:, 1]) & (indices[:, 1] != 0)
    i = indices[mask, 0]
    j = indices[mask, 1]
    
    ratios = np.array([np.linalg.norm(Xr[i] - Xr[j]) / np.linalg.norm(X[i] - X[j]) for i,j in zip(indices[mask, 0], indices[mask, 1])])

    result["X_reduced"] = Xr
    result["mean_ratio"] = ratios.mean() if len(ratios) else np.nan
    result["max_distortion"] = np.max(np.abs(ratios - 1))

    return result

# Step 11 - unroll_swiss_roll
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

def unroll_swiss_roll(n_samples=1000, random_state=42):
    # make_swiss_roll + LocallyLinearEmbedding(2, n_neighbors=10); report correlation with t.
    X, t = make_swiss_roll(
        n_samples=n_samples, 
        noise=0.2, 
        random_state=random_state
    )

    X_2d = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=random_state).fit_transform(X)

    return {
        "X_2d": X_2d,
        "t_correlation": float(np.max(np.maximum(
            np.abs(np.corrcoef(t, X_2d[:, 0])),
            np.abs(np.corrcoef(t, X_2d[:, 1]))
        )))
    }

# Step 12 - tsne_map
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist

def tsne_map(X, y, n=1000, random_state=42):
    # TSNE(2, init='pca'); return X_2d and a between/within class separation ratio.
    X_2d = TSNE(n_components=2, init="pca", random_state=random_state).fit_transform(X[:n])
    y_new = y[:n]

    classes = np.unique(y_new)
    centroids = [np.mean(X_2d[y_new == i], axis= 0) for i in classes]

    within_class_distances = np.array([np.linalg.norm(X_2d[y_new == c] - centroids[c], axis=1).mean() for c in classes])

    return {
        "X_2d": X_2d,
        "separation": np.mean(pdist(centroids)) / np.mean(within_class_distances)
    }

# Step 13 - classifier_on_compressed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def classifier_on_compressed(X, y, n_components, test_size=0.25, random_state=42):
    # raw vs PCA-feature logistic regression on a held-out split; PCA fitted on the training split only.
    result = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logReg = LogisticRegression(max_iter=1000)
    logReg.fit(X_train, y_train)
    result["raw_accuracy"] = logReg.score(X_test, y_test)

    pca = fit_pca(X_train, n_components, random_state)
    pca_train = pca.transform(X_train)
    pca_test = pca.transform(X_test)
    logReg.fit(pca_train, y_train)
    result["pca_accuracy"] = logReg.score(pca_test, y_test)

    result["n_features"] = (X_train.shape[1], n_components)

    return result

# Step 14 - pca_classifier_pipeline (not yet solved)
# TODO: implement

# Step 15 - save_and_reload_pipeline (not yet solved)
# TODO: implement

# Step 16 - predict_images (not yet solved)
# TODO: implement

