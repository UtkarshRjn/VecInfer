"""VecInfer bit-width presets and codebook checks.

Paper config ``d4b8`` is a sub-vector of length 4 with an 8-bit codebook.
Head dimension is 128, so the number of sub-quantizers is ``128 / 4 = 32``.
Bits per KV element are ``M * nbits / d``.
"""

from __future__ import annotations

from pathlib import Path

HEAD_DIM = 128
N_LAYERS = 32

# sub-vector length, codebook bits. M = HEAD_DIM / sub-vector length.
PRESETS: dict[str, dict[str, int]] = {
    "d8b8": {"subvector": 8, "nbits": 8},    # 1 bit
    "d8b12": {"subvector": 8, "nbits": 12},  # 1.5 bit
    "d4b8": {"subvector": 4, "nbits": 8},    # 2 bit, VecInfer Table 2
    "d4b12": {"subvector": 4, "nbits": 12},  # 3 bit
    "d2b8": {"subvector": 2, "nbits": 8},    # 4 bit
}


def preset_to_pq(name: str) -> tuple[int, int]:
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown VecInfer preset {name!r}. Known presets: {known}")
    spec = PRESETS[name]
    subvector = spec["subvector"]
    if HEAD_DIM % subvector != 0:
        raise ValueError(f"Head dim {HEAD_DIM} is not divisible by sub-vector {subvector}")
    return HEAD_DIM // subvector, spec["nbits"]


def bits_per_dimension(m_subquantizers: int, nbits: int, head_dim: int = HEAD_DIM) -> float:
    if m_subquantizers <= 0 or nbits <= 0 or head_dim <= 0:
        raise ValueError("M, nbits, and head dim must be positive")
    if head_dim % m_subquantizers != 0:
        raise ValueError(f"Head dim {head_dim} is not divisible by M={m_subquantizers}")
    return (m_subquantizers * nbits) / head_dim


def codebook_path(root: Path, kind: str, layer: int, m_subquantizers: int, nbits: int) -> Path:
    return root / f"{kind}_cent_layer{layer}_{m_subquantizers}_{nbits}.pq.pt"


def missing_codebooks(
    root: Path,
    m_subquantizers: int,
    nbits: int,
    n_layers: int = N_LAYERS,
) -> list[Path]:
    missing: list[Path] = []
    for layer in range(n_layers):
        for kind in ("key", "val"):
            path = codebook_path(root, kind, layer, m_subquantizers, nbits)
            if not path.is_file():
                missing.append(path)
    return missing
