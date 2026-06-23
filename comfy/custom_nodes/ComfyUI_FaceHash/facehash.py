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


# --- arcface_blend_v2 — ON-MANIFOLD hash (blend toward a key-selected real identity) ---
# Keep _keyed_target / _slerp / arcface_blend_v2 byte-identical to local/arcface_hash.py so the
# InstantID texture path and the MICA geometry path select the SAME gallery person under one key.
def _keyed_target(key: str, n_gallery: int, n_mix: int = 2):
    seed = int.from_bytes(hashlib.sha256(("blend:" + key).encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n_mix = max(1, min(int(n_mix), int(n_gallery)))
    idx = rng.choice(n_gallery, size=n_mix, replace=False)
    w = rng.dirichlet(np.ones(n_mix)) if n_mix > 1 else np.array([1.0])
    return idx, w


def _slerp(a, b, t):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b)
    dot = np.clip(a @ b, -1.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    safe = np.where(sin < 1e-6, 1.0, sin)
    ca = np.where(sin < 1e-6, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    cb = np.where(sin < 1e-6, t, np.sin(t * theta) / safe)
    out = ca[..., None] * a + cb[..., None] * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def arcface_blend_v2(embedding, key: str, gallery, strength: float = 0.6, n_mix: int = 2):
    """Blend the input embedding toward a key-selected real identity from `gallery` (slerp),
    preserving norm. gallery: (G,D) real embeddings in the SAME space as embedding."""
    e = np.asarray(embedding, dtype=np.float64)
    G = np.asarray(gallery, dtype=np.float64)
    if G.ndim != 2 or G.shape[-1] != e.shape[-1]:
        raise ValueError(f"gallery {G.shape} incompatible with embedding dim {e.shape[-1]}")
    nrm = np.linalg.norm(e, axis=-1, keepdims=True)
    Gn = G / np.linalg.norm(G, axis=-1, keepdims=True)
    idx, w = _keyed_target(key, len(Gn), n_mix)
    target = (w[:, None] * Gn[idx]).sum(0)
    target = target / np.linalg.norm(target)
    out = _slerp(e, target, float(strength)) * nrm
    return out.astype(np.float32)


def load_gallery(path, column: str = "antelope"):
    """Load a paired gallery .npz (row-aligned 'antelope'/'mica') or a plain (G,D) .npy."""
    p = str(path)
    if p.endswith(".npz"):
        z = np.load(p)
        if column not in z:
            raise KeyError(f"gallery {p} has no column {column!r}; has {list(z.keys())}")
        return z[column].astype(np.float64)
    return np.load(p).astype(np.float64)


# --- arcface_keymix_whitened_v3 — ON-MANIFOLD via WHITENING (derived synthetic id, no targeting) ---
# Keep byte-identical to local/arcface_hash.py. keymix in the whitened PCA-identity subspace, where a
# permutation preserves the (now isotropic) distribution -> stays in-distribution -> plausible synth.
def arcface_keymix_whitened_v3(embedding, key: str, basis, offset: float = 0.0):
    """basis = (mu (D,), V (k,D) PCA rows, sigma (k,) per-axis std) from build_basis.py."""
    e = np.asarray(embedding, dtype=np.float64)
    mu, V, sigma = (np.asarray(b, dtype=np.float64) for b in basis)
    k = V.shape[0]
    nrm = np.linalg.norm(e, axis=-1, keepdims=True)
    eu = e / nrm                                         # scale-invariant: raw or normed input
    z = ((eu - mu) @ V.T) / sigma
    perm, signs, rng = _keyed(key, k)
    zt = signs * z[..., perm]
    if offset:
        zt = zt + offset * rng.standard_normal(zt.shape)
    out = mu + (zt * sigma) @ V
    out = out / np.linalg.norm(out, axis=-1, keepdims=True) * nrm
    return out.astype(np.float32)


def load_basis(path, column: str = "antelope"):
    """Load a whitening basis .npz ('<col>_mu','<col>_V','<col>_sigma'). -> (mu, V (k,D), sigma)."""
    z = np.load(str(path))
    pre = column + "_"
    miss = [pre + n for n in ("mu", "V", "sigma") if pre + n not in z]
    if miss:
        raise KeyError(f"basis {path} missing {miss}; has {list(z.keys())}")
    return z[pre + "mu"].astype(np.float64), z[pre + "V"].astype(np.float64), z[pre + "sigma"].astype(np.float64)
