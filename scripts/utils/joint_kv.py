"""Train one product-quantization codebook on keys and values together.

The separate baseline fits one k-means codebook on key sub-vectors and another
on value sub-vectors. The joint codebook is fit on the pooled sub-vectors and
then used for both. Both use the same number of centroids and the same
sub-vector width, so the bit rate matches and only the reconstruction error
changes.
"""

import numpy as np


def _as_f32(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected a (n, d) array, got {X.shape}")
    return np.ascontiguousarray(X)


def subvectors(X, M):
    X = _as_f32(X)
    n, d = X.shape
    if d % M != 0:
        raise ValueError(f"d={d} is not divisible by M={M}")
    return X.reshape(-1, d // M)


def _kmeans_faiss(X, k, niter):
    import faiss
    kmeans = faiss.Kmeans(d=X.shape[1], k=k, niter=niter, verbose=False)
    kmeans.seed = 42
    kmeans.train(np.ascontiguousarray(X, dtype=np.float32))
    return np.ascontiguousarray(kmeans.centroids, dtype=np.float32)


def _kmeans_numpy(X, k, niter, seed=42):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
    labels = np.empty(X.shape[0], dtype=np.int32)
    for _ in range(niter):
        centroid_norm = np.sum(centroids * centroids, axis=1)
        for start in range(0, X.shape[0], 8192):
            block = X[start:start + 8192]
            dist = (
                np.sum(block * block, axis=1, keepdims=True)
                + centroid_norm
                - 2.0 * (block @ centroids.T)
            )
            labels[start:start + block.shape[0]] = np.argmin(dist, axis=1)
        updated = centroids.copy()
        for j in range(k):
            chosen = labels == j
            if np.any(chosen):
                updated[j] = X[chosen].mean(axis=0)
        shift = np.max(np.abs(updated - centroids))
        centroids = updated
        if shift < 1e-4:
            break
    return centroids


def train_codebook(X, M, nbits, niter=30, seed=42):
    """Return centroids of shape (1, 2**nbits, d/M)."""
    blocks = subvectors(X, M)
    k = 2 ** nbits
    if blocks.shape[0] < k:
        raise ValueError(f"need at least {k} sub-vectors, got {blocks.shape[0]}")
    try:
        centroids = _kmeans_faiss(blocks, k, niter)
    except ImportError:
        centroids = _kmeans_numpy(blocks, k, niter, seed=seed)
    return centroids.reshape(1, k, blocks.shape[1])


def reconstruction_mse(X, centroids, M):
    """Mean squared error per coordinate after nearest-centroid assignment."""
    blocks = subvectors(X, M)
    cents = np.ascontiguousarray(centroids.reshape(-1, blocks.shape[1]), dtype=np.float32)
    centroid_norm = np.sum(cents * cents, axis=1)
    total = 0.0
    for start in range(0, blocks.shape[0], 8192):
        block = blocks[start:start + 8192]
        dist = (
            np.sum(block * block, axis=1, keepdims=True)
            + centroid_norm
            - 2.0 * (block @ cents.T)
        )
        total += float(np.min(dist, axis=1).sum())
    return total / blocks.size


def compare_separate_and_joint(key, value, M, nbits, niter=30, seed=42):
    """Fit separate and joint codebooks and report reconstruction MSE.

    ``key`` and ``value`` are (n, d) arrays of head vectors, the same layout
    written to the sampled ``*.fvecs`` files. ``M`` and ``nbits`` must match
    for the two streams so the joint codebook has the same rate.
    """
    key = _as_f32(key)
    value = _as_f32(value)
    if key.shape[1] != value.shape[1]:
        raise ValueError(f"key dim {key.shape[1]} != value dim {value.shape[1]}")

    key_cent = train_codebook(key, M, nbits, niter=niter, seed=seed)
    value_cent = train_codebook(value, M, nbits, niter=niter, seed=seed + 1)
    joint_cent = train_codebook(
        np.concatenate([key, value], axis=0), M, nbits, niter=niter, seed=seed + 2
    )

    separate_key = reconstruction_mse(key, key_cent, M)
    separate_value = reconstruction_mse(value, value_cent, M)
    joint_key = reconstruction_mse(key, joint_cent, M)
    joint_value = reconstruction_mse(value, joint_cent, M)
    separate_mean = 0.5 * (separate_key + separate_value)
    joint_mean = 0.5 * (joint_key + joint_value)
    return {
        "separate_key_mse": separate_key,
        "separate_value_mse": separate_value,
        "separate_mean_mse": separate_mean,
        "joint_key_mse": joint_key,
        "joint_value_mse": joint_value,
        "joint_mean_mse": joint_mean,
        "mean_mse_change": joint_mean - separate_mean,
        "mean_mse_ratio": joint_mean / separate_mean,
        "key_cent": key_cent,
        "value_cent": value_cent,
        "joint_cent": joint_cent,
    }
