"""Fit product-quantization centroids to attention-output error.

VecInfer initializes a codebook by k-means on key blocks and another on value
blocks. This module keeps that parameterization, 256 centroids of width 4, and
moves the centroids to reduce the gap in the causal attention output.
"""

import torch
import torch.nn.functional as F


def kmeans(x, k, niter=20, seed=42):
    """Lloyd k-means. ``x`` is (n, d) and the result is (k, d)."""
    if x.shape[0] < k:
        raise ValueError(f"need at least {k} rows, got {x.shape[0]}")
    generator = torch.Generator().manual_seed(seed)
    choice = torch.randperm(x.shape[0], generator=generator)[:k]
    centroids = x[choice.to(x.device)].clone()
    for _ in range(niter):
        distance = (
            x.pow(2).sum(-1, keepdim=True)
            + centroids.pow(2).sum(-1)
            - 2.0 * (x @ centroids.T)
        )
        assignment = distance.argmin(dim=-1)
        counts = torch.bincount(assignment, minlength=k)
        updated = torch.zeros_like(centroids)
        updated.index_add_(0, assignment, x)
        occupied = counts > 0
        centroids = centroids.clone()
        centroids[occupied] = updated[occupied] / counts[occupied].unsqueeze(1).to(x.dtype)
    return centroids


def quantize_ste(x, centroids, block_size):
    """Nearest-centroid quantization.

    ``x`` is (..., d). One centroid table is shared by every block, matching
    VecInfer. The index is a hard lookup. Gradients flow into the chosen centroid.
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"d={x.shape[-1]} is not divisible by block size {block_size}")
    blocks = x.reshape(-1, block_size)
    distance = (
        blocks.pow(2).sum(-1, keepdim=True)
        + centroids.pow(2).sum(-1)
        - 2.0 * (blocks @ centroids.T)
    )
    # Hard index, with gradient flowing into the selected centroid.
    reconstruction = centroids[distance.argmin(dim=-1)]
    return reconstruction.reshape(x.shape)


def reconstruction_mse(x, centroids, block_size):
    quantized = quantize_ste(x, centroids, block_size)
    return (quantized - x).pow(2).mean()


def attention_output(q, k, v):
    """Causal grouped-query attention. ``q`` is (Hq, T, d), ``k`` and ``v`` are (Hkv, T, d)."""
    groups = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(groups, dim=0).contiguous()
    v = v.repeat_interleave(groups, dim=0).contiguous()
    return F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True
    ).squeeze(0)


def output_mse(q, k, v, k_centroids, v_centroids, block_size, target):
    k_hat = quantize_ste(k, k_centroids, block_size)
    v_hat = quantize_ste(v, v_centroids, block_size)
    prediction = attention_output(q, k_hat, v_hat)
    return (prediction - target).pow(2).mean()


def fit_attention_codebooks(
    q, k, v, block_size=4, nbits=8, kmeans_iters=20, steps=40, lr=1e-2, seed=42
):
    """Minimize causal attention-output error, starting from k-means centroids.

    ``q`` is (Hq, T, d). ``k`` and ``v`` are (Hkv, T, d). Keys must already be
    the vectors that enter the cache (scaled and Hadamard-rotated). Values are
    the raw projections. Both codebooks keep ``2**nbits`` centroids.
    """
    d = k.shape[-1]
    k_rows = k.reshape(-1, d).float()
    v_rows = v.reshape(-1, d).float()
    q = q.float()
    k = k.float()
    v = v.float()
    k_count = 2 ** nbits

    k_init = kmeans(k_rows.reshape(-1, block_size), k_count, niter=kmeans_iters, seed=seed)
    v_init = kmeans(v_rows.reshape(-1, block_size), k_count, niter=kmeans_iters, seed=seed + 1)
    with torch.no_grad():
        target = attention_output(q, k, v)
        baseline = output_mse(q, k, v, k_init, v_init, block_size, target)
        baseline_key = reconstruction_mse(k_rows, k_init, block_size)
        baseline_value = reconstruction_mse(v_rows, v_init, block_size)

    k_centroids = torch.nn.Parameter(k_init.clone())
    v_centroids = torch.nn.Parameter(v_init.clone())
    optimizer = torch.optim.Adam([k_centroids, v_centroids], lr=lr)
    best_loss = baseline.detach()
    best_k = k_centroids.detach().clone()
    best_v = v_centroids.detach().clone()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = output_mse(q, k, v, k_centroids, v_centroids, block_size, target)
        loss.backward()
        optimizer.step()
        if torch.isfinite(loss) and loss.detach() < best_loss:
            best_loss = loss.detach()
            best_k = k_centroids.detach().clone()
            best_v = v_centroids.detach().clone()

    with torch.no_grad():
        fitted_key = reconstruction_mse(k_rows, best_k, block_size)
        fitted_value = reconstruction_mse(v_rows, best_v, block_size)
    return {
        "kmeans_output_mse": float(baseline),
        "fitted_output_mse": float(best_loss),
        "output_mse_ratio": float(best_loss / baseline),
        "kmeans_key_mse": float(baseline_key),
        "fitted_key_mse": float(fitted_key),
        "kmeans_value_mse": float(baseline_value),
        "fitted_value_mse": float(fitted_value),
    }
