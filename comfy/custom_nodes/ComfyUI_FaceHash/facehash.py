"""Vendored copy of the Stage-2 ArcFace hash, for the ComfyUI node.

Keep byte-for-byte in sync with ``local/arcface_hash.py`` (same SHA-256 seed
derivation -> same permutation/signs -> the node renders the same hashed face
that the local pipeline computes). numpy-only; the node bridges torch<->numpy.
"""

import hashlib

import numpy as np


def _keyed(key: str, n: int):
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    return perm, signs, rng


def arcface_keymix_v1(embedding, key: str, offset: float = 0.0):
    """Signed permutation (orthogonal) of a 512-d ArcFace embedding / (N,512) batch."""
    e = np.asarray(embedding, dtype=np.float64)
    n = e.shape[-1]
    perm, signs, rng = _keyed(key, n)
    out = signs * e[..., perm]
    if offset:
        nrm = np.linalg.norm(e, axis=-1, keepdims=True)
        out = out + offset * rng.standard_normal(out.shape)
        out = out / np.linalg.norm(out, axis=-1, keepdims=True) * nrm
    return out.astype(np.float32)


def arcface_unmix_v1(hashed, key: str):
    h = np.asarray(hashed, dtype=np.float64)
    n = h.shape[-1]
    perm, signs, _ = _keyed(key, n)
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    return (h[..., inv] * signs[inv]).astype(np.float32)
