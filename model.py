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

# Step 9 - incremental_pca (not yet solved)
# TODO: implement

# Step 10 - random_projection (not yet solved)
# TODO: implement

# Step 11 - unroll_swiss_roll (not yet solved)
# TODO: implement

# Step 12 - tsne_map (not yet solved)
# TODO: implement

# Step 13 - classifier_on_compressed (not yet solved)
# TODO: implement

# Step 14 - pca_classifier_pipeline (not yet solved)
# TODO: implement

# Step 15 - save_and_reload_pipeline (not yet solved)
# TODO: implement

# Step 16 - predict_images (not yet solved)
# TODO: implement

