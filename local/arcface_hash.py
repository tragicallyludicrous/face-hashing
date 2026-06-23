"""arcface_hash.py — Stage-2 identity hash, in ArcFace space.

    transform(embedding, key) -> embedding'      # the "hash function"

Strategy ``arcface_keymix_v1``: a key-seeded SIGNED PERMUTATION of the 512-d
ArcFace identity vector (optionally plus a small bounded offset). A signed
permutation matrix is *orthogonal*, which is exactly what we want for ArcFace —
the embedding lives (up to scale) on a hypersphere and identity is compared by
cosine. So the transform:

  * preserves the vector's norm                       (stays on the sphere),
  * preserves pairwise geometry under a fixed key     (it's an isometry: two
    photos of one person stay close; two people stay apart — so the hashed
    space is still a coherent identity space),
  * sends each identity to cosine ~0 vs its own original (a *different* person
    to any recognizer), and
  * is deterministic, face-dependent, and exactly reversible WITH the key
    (offset=0); obfuscation, not encryption.

This is the ArcFace-space analog of the FLAME-shape ``flame_shape_keymix_v1``
used in the studio viewer. Here it is the Stage-2 step that feeds Stage-4
(InstantID): the rendered face becomes a deterministic function of the *input
identity*, not of the diffusion seed. Pose/expression are carried separately
(InstantID keypoints / SMIRK), so only identity is mutated.

NOTE: keep ``arcface_keymix_v1`` byte-for-byte in sync with the vendored copy in
``comfy/custom_nodes/ComfyUI_FaceHash/facehash.py`` (the ComfyUI node that
injects this into InstantID). Same seed derivation -> same permutation -> same
face on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #
def _keyed(key: str, n: int):
    """Derive the (permutation, signs, rng) for a key. Deterministic & stable
    across machines: seed = first 8 bytes of SHA-256(key), little-endian."""
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    return perm, signs, rng  # rng returned so offset noise draws AFTER perm/signs


def arcface_keymix_v1(embedding, key: str, offset: float = 0.0):
    """Hash a 512-d ArcFace embedding (or a (N,512) batch) under ``key``.

    offset=0 -> pure signed permutation (orthogonal, exactly reversible).
    offset>0 -> add bounded Gaussian noise and renormalize to the original norm
                (more scramble, only *approximately* reversible)."""
    e = np.asarray(embedding, dtype=np.float64)
    n = e.shape[-1]
    perm, signs, rng = _keyed(key, n)

    out = signs * e[..., perm]                      # signed permutation == orthogonal
    if offset:
        nrm = np.linalg.norm(e, axis=-1, keepdims=True)
        out = out + offset * rng.standard_normal(out.shape)
        out = out / np.linalg.norm(out, axis=-1, keepdims=True) * nrm
    return out.astype(np.float32)


def arcface_unmix_v1(hashed, key: str):
    """Invert the signed permutation (exact for offset=0). Recovers the original
    embedding up to the offset noise when offset>0."""
    h = np.asarray(hashed, dtype=np.float64)
    n = h.shape[-1]
    perm, signs, _ = _keyed(key, n)
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    return (h[..., inv] * signs[inv]).astype(np.float32)


# --------------------------------------------------------------------------- #
# arcface_blend_v2 — ON-MANIFOLD identity hash (blend toward a key-selected real id)
# --------------------------------------------------------------------------- #
# Why this exists: arcface_keymix_v1 is a signed PERMUTATION — orthogonal, so it preserves
# norm and the *recognition* geometry (intra/inter cosine), but it maps the real-face
# manifold onto a DIFFERENT, permuted submanifold. A permuted real embedding is therefore
# out-of-distribution for a *generator* (InstantID), which only ever saw on-manifold
# embeddings in training -> some inputs render a plausible different person, others render a
# monster. blend_v2 instead moves the input toward a REAL identity the key selects from a
# gallery; a blend of two real faces stays near the manifold -> plausible for any input.
# It still depends on the *input* identity (so it's a hash, not just "key picks a face"),
# is deterministic, key-driven, and obfuscating (uninvertible without the gallery + key).


def _keyed_target(key: str, n_gallery: int, n_mix: int = 2):
    """Deterministic virtual-identity pick from a size-`n_gallery` gallery: choose `n_mix`
    rows + convex (Dirichlet) weights from the key's RNG. Cross-machine stable; MUST stay
    byte-identical to the vendored copy so geometry (MICA) and texture (InstantID) pick the
    SAME gallery person from their respective (row-aligned) galleries under one key."""
    seed = int.from_bytes(hashlib.sha256(("blend:" + key).encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n_mix = max(1, min(int(n_mix), int(n_gallery)))
    idx = rng.choice(n_gallery, size=n_mix, replace=False)
    w = rng.dirichlet(np.ones(n_mix)) if n_mix > 1 else np.array([1.0])
    return idx, w


def _slerp(a, b, t):
    """Spherical interpolation on the unit sphere. a: (...,D); b: (D,); t scalar in [0,1].
    Returns unit (...,D). Falls back to lerp where the two are nearly colinear."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b)
    dot = np.clip(a @ b, -1.0, 1.0)                       # (...,)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    safe = np.where(sin < 1e-6, 1.0, sin)
    ca = np.where(sin < 1e-6, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    cb = np.where(sin < 1e-6, t, np.sin(t * theta) / safe)
    out = ca[..., None] * a + cb[..., None] * b           # cb*b broadcasts (D,) over (...,)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def arcface_blend_v2(embedding, key: str, gallery, strength: float = 0.6, n_mix: int = 2):
    """On-manifold identity hash: blend the input embedding toward a key-selected real
    identity from `gallery` (slerp on the sphere), preserving the input's norm.

    embedding : (D,) or (N,D) raw ArcFace embedding, SAME backbone/space as `gallery`.
    gallery   : (G,D) real embeddings in that space (see build_gallery.py).
    strength  : 0 -> original identity, 1 -> the key's virtual target. ~0.5-0.8 typical.
    n_mix     : # gallery rows convex-mixed into the virtual target (richer id space).
    Deterministic in (embedding, key, gallery, strength, n_mix)."""
    e = np.asarray(embedding, dtype=np.float64)
    G = np.asarray(gallery, dtype=np.float64)
    if G.ndim != 2 or G.shape[-1] != e.shape[-1]:
        raise ValueError(f"gallery {G.shape} incompatible with embedding dim {e.shape[-1]}")
    nrm = np.linalg.norm(e, axis=-1, keepdims=True)          # keep the input's scale
    Gn = G / np.linalg.norm(G, axis=-1, keepdims=True)
    idx, w = _keyed_target(key, len(Gn), n_mix)
    target = (w[:, None] * Gn[idx]).sum(0)
    target = target / np.linalg.norm(target)                 # the virtual identity (unit)
    out = _slerp(e, target, float(strength)) * nrm
    return out.astype(np.float32)


def load_gallery(path, column: str = "antelope"):
    """Load a paired gallery: a `.npz` with row-aligned columns 'antelope' (InstantID) and
    'mica' (depth), or a plain (G,D) `.npy`. Returns the requested column as (G,D) float64."""
    p = str(path)
    if p.endswith(".npz"):
        z = np.load(p)
        if column not in z:
            raise KeyError(f"gallery {p} has no column {column!r}; has {list(z.keys())}")
        return z[column].astype(np.float64)
    return np.load(p).astype(np.float64)


# --------------------------------------------------------------------------- #
# arcface_keymix_whitened_v3 — ON-MANIFOLD via WHITENING (derived synthetic id, no targeting)
# --------------------------------------------------------------------------- #
# keymix_v1 is off-manifold because the real-embedding cloud is *anisotropic* (correlated dims,
# very different per-direction variances) — a permutation of raw coords lands outside it. But a
# permutation is distribution-preserving when the coords are ISOTROPIC. So: whiten the identity
# subspace (mean + PCA + per-axis std, learned ONCE from a corpus of *synthetic* faces — people who
# don't exist), permute THERE, then un-whiten. The scrambled point stays inside the real-embedding
# distribution -> a plausible but fully DERIVED synthetic identity (no real face targeted). The
# "basis" is aggregate statistics (mu, V, sigma); no faces are retained. See build_basis.py.


def arcface_keymix_whitened_v3(embedding, key: str, basis, offset: float = 0.0):
    """Identity hash by keymix in the whitened PCA-identity subspace.

    embedding : (D,) or (N,D) raw ArcFace embedding (same backbone as the basis).
    basis     : (mu (D,), V (k,D) PCA rows, sigma (k,) per-axis std) from build_basis.py.
    offset    : >0 adds isotropic Gaussian in whitened space (still in-distribution) for more
                scramble; 0 = pure permutation (subspace-reversible with the key)."""
    e = np.asarray(embedding, dtype=np.float64)
    mu, V, sigma = (np.asarray(b, dtype=np.float64) for b in basis)
    k = V.shape[0]
    nrm = np.linalg.norm(e, axis=-1, keepdims=True)
    eu = e / nrm                                         # unit space (basis is unit) -> scale-invariant:
    z = ((eu - mu) @ V.T) / sigma                        # works on raw (InstantID) or normed embeddings
    perm, signs, rng = _keyed(key, k)
    zt = signs * z[..., perm]                            # permutation preserves an isotropic dist
    if offset:
        zt = zt + offset * rng.standard_normal(zt.shape)
    out = mu + (zt * sigma) @ V                          # un-whiten + reconstruct (drops residual)
    out = out / np.linalg.norm(out, axis=-1, keepdims=True) * nrm
    return out.astype(np.float32)


def load_basis(path, column: str = "antelope"):
    """Load a whitening basis .npz (per-column '<col>_mu','<col>_V','<col>_sigma' from
    build_basis.py). Returns (mu (D,), V (k,D), sigma (k,))."""
    z = np.load(str(path))
    pre = column + "_"
    miss = [pre + n for n in ("mu", "V", "sigma") if pre + n not in z]
    if miss:
        raise KeyError(f"basis {path} missing {miss}; has {list(z.keys())}")
    return z[pre + "mu"].astype(np.float64), z[pre + "V"].astype(np.float64), z[pre + "sigma"].astype(np.float64)


# registry hook (Stage-2 plan: hot-swappable strategies)
TRANSFORMS = {
    "arcface_keymix_v1": arcface_keymix_v1,
    "arcface_blend_v2": arcface_blend_v2,
    "arcface_keymix_whitened_v3": arcface_keymix_whitened_v3,
}


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# self-test / demonstration
# --------------------------------------------------------------------------- #
def selftest(key: str = "demo-key"):
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.linalg.norm(v)

    e = unit(rng.standard_normal(512))                  # an identity
    e_same = unit(e + 0.15 * rng.standard_normal(512))  # another photo, same person (close)
    e_other = unit(rng.standard_normal(512))            # a different person

    h = arcface_keymix_v1(e, key)
    h2 = arcface_keymix_v1(e, key)                       # determinism
    h_same = arcface_keymix_v1(e_same, key)
    h_other = arcface_keymix_v1(e_other, key)
    h_k2 = arcface_keymix_v1(e, "another-key")           # key dependence
    rec = arcface_unmix_v1(h, key)                       # reversibility

    print(f"  key = {key!r}")
    print("  deterministic (same in twice)      :", np.allclose(h, h2))
    print(f"  norm preserved   ||e||={np.linalg.norm(e):.4f}  ->  ||h||={np.linalg.norm(h):.4f}")
    print(f"  cos(e, hash(e))                    : {cosine(e, h):+.4f}   (want ~0: different identity)")
    print(f"  cos(e, hash_otherkey(e))           : {cosine(e, h_k2):+.4f}   (want ~0: key-dependent)")
    print(f"  cos(hash_k1, hash_k2) same face    : {cosine(h, h_k2):+.4f}   (want ~0: keys decorrelate)")
    print(f"  reversibility cos(e, unmix(hash))  : {cosine(e, rec):+.4f}   (want 1.0)")
    print("  --- isometry: structure preserved under the key ---")
    print(f"  cos(e, e_same)  {cosine(e, e_same):+.4f}  ->  cos(hash,hash) {cosine(h, h_same):+.4f}   (same person stays close)")
    print(f"  cos(e, e_other) {cosine(e, e_other):+.4f}  ->  cos(hash,hash) {cosine(h, h_other):+.4f}   (different stays far)")

    offs = arcface_keymix_v1(e, key, offset=0.3)
    print(f"  with offset=0.3: cos(e,hash) {cosine(e, offs):+.4f}, ||h||={np.linalg.norm(offs):.4f}")

    # --- arcface_blend_v2: on-manifold (blend toward a key-selected real identity) ---
    print("  --- arcface_blend_v2 (gallery blend) ---")
    gal = rng.standard_normal((64, 512))
    gal = gal / np.linalg.norm(gal, axis=1, keepdims=True)        # a stand-in "real-id gallery"
    bm = lambda v, k, s=0.6: arcface_blend_v2(v, k, gal, strength=s)
    b1, b2 = bm(e, key), bm(e, key)
    print("  deterministic (same in twice)      :", np.allclose(b1, b2))
    print(f"  norm preserved   ||e||={np.linalg.norm(e):.4f}  ->  ||b||={np.linalg.norm(b1):.4f}")
    print(f"  cos(e, blend) strength 0.3/0.6/0.9 : "
          f"{cosine(e, bm(e, key, 0.3)):+.3f} / {cosine(e, bm(e, key, 0.6)):+.3f} / {cosine(e, bm(e, key, 0.9)):+.3f}"
          "   (input-dependent hash, decreasing)")
    print(f"  cos(e, blend_otherkey)             : {cosine(e, bm(e, 'another-key')):+.3f}   (key-dependent)")
    print(f"  same person stays close  cos(b,b_same) {cosine(bm(e, key), bm(e_same, key)):+.3f}  "
          f"(orig {cosine(e, e_same):+.3f})")
    # on-manifold check: nearest-gallery cosine, original vs blended (blend should sit nearer the gallery)
    near = lambda v: float(np.max((gal @ (v / np.linalg.norm(v)))))
    print(f"  nearest-gallery cos  e={near(e):+.3f} -> blend(0.9)={near(bm(e, key, 0.9)):+.3f}   "
          "(blend pulled toward the real-id set)")

    # --- arcface_keymix_whitened_v3: on-manifold via whitening (DERIVED synthetic, no targeting) ---
    print("  --- arcface_keymix_whitened_v3 (whitened keymix) ---")
    kdim = 40
    Vtrue = np.linalg.qr(rng.standard_normal((512, kdim)))[0].T          # (k,512) true id axes
    sig_true = np.linspace(1.0, 0.2, kdim)                               # anisotropic variances

    def sample(n):                                                      # a synthetic "id manifold"
        X = (rng.standard_normal((n, kdim)) * sig_true) @ Vtrue + 0.02 * rng.standard_normal((n, 512))
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    E = sample(4000)
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    basis = (mu, Vt[:kdim], S[:kdim] / np.sqrt(len(E) - 1))
    test = sample(200)                                                  # held-out on-manifold faces

    def off(M):  # fraction of energy OUTSIDE the id subspace: ~0 on-manifold, ->1 off-manifold
        r = M - basis[0]
        proj = (r @ basis[1].T) @ basis[1]
        return float(np.mean(np.sum((r - proj) ** 2, axis=1) / np.sum(r ** 2, axis=1)))

    v3 = np.stack([arcface_keymix_whitened_v3(t, key, basis) for t in test])
    rawk = np.stack([arcface_keymix_v1(t, key) for t in test])
    print(f"  off-manifold energy fraction (0=on-manifold, ->1=off):")
    print(f"     on-manifold {off(test):.2f}   raw keymix {off(rawk):.2f}   v3 {off(v3):.2f}")
    print("  deterministic:", np.allclose(arcface_keymix_whitened_v3(test[0], key, basis),
                                           arcface_keymix_whitened_v3(test[0], key, basis)))
    print(f"  cos(e, v3) same/other key : {cosine(test[0], v3[0]):+.3f} / "
          f"{cosine(test[0], arcface_keymix_whitened_v3(test[0], 'k2', basis)):+.3f}   (key-dependent)")


def compare(folder: str, key: str):
    """Across a folder of <id>-<...>_arcface.npy, show that hashing moves every
    identity far from its original while preserving intra/inter structure."""
    paths = sorted(pathlib.Path(folder).rglob("*arcface.npy"))
    if not paths:
        print("no *arcface.npy under", folder)
        return
    embs, ids = [], []
    for p in paths:
        embs.append(np.load(p))
        ids.append(p.stem.split("-")[0].split("_")[0].lower())
    E = np.stack(embs)
    H = arcface_keymix_v1(E, key)
    ids = np.array(ids)

    def pair_stats(M):
        n = len(M)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        C = Mn @ Mn.T
        same = np.zeros((n, n), bool)
        for u in np.unique(ids):
            idx = np.where(ids == u)[0]
            same[np.ix_(idx, idx)] = True
        np.fill_diagonal(same, False)
        off = ~np.eye(n, dtype=bool)
        intra = C[same].mean() if same.any() else float("nan")
        inter = C[off & ~same].mean()
        return intra, inter

    oi, oe = pair_stats(E)
    hi, he = pair_stats(H)
    selfcos = np.mean([cosine(E[i], H[i]) for i in range(len(E))])
    print(f"  {len(E)} embeddings, {len(np.unique(ids))} identities, key={key!r}")
    print(f"  mean cos(original_i, hashed_i)     : {selfcos:+.4f}   (want ~0: you are now someone else)")
    print(f"  intra-identity cosine  original->hashed : {oi:+.4f} -> {hi:+.4f}")
    print(f"  inter-identity cosine  original->hashed : {oe:+.4f} -> {he:+.4f}")
    print("  (intra/inter gap is preserved -> hashed space is still a valid identity space)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stage-2 ArcFace identity hash (arcface_keymix_v1).")
    ap.add_argument("input", nargs="?", help="an arcface.npy, or a dir of them")
    ap.add_argument("-o", "--output", help="output .npy (file) or dir")
    ap.add_argument("-k", "--key", default="demo-key")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--transform", choices=list(TRANSFORMS), default="arcface_keymix_v1")
    ap.add_argument("--gallery", help="paired gallery .npz/.npy (required for arcface_blend_v2)")
    ap.add_argument("--basis", help="whitening basis .npz (required for arcface_keymix_whitened_v3)")
    ap.add_argument("--column", default="antelope", help="gallery/basis column: antelope|mica")
    ap.add_argument("--strength", type=float, default=0.6, help="blend_v2: 0=self .. 1=target id")
    ap.add_argument("--n-mix", type=int, default=2, help="blend_v2: # gallery rows mixed per key")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", metavar="DIR", help="show intra/inter cosine before/after hashing")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.key)
        return
    if args.compare:
        compare(args.compare, args.key)
        return
    if not args.input:
        ap.error("give an input .npy/dir, or --selftest / --compare DIR")

    if args.transform == "arcface_blend_v2":
        if not args.gallery:
            ap.error("arcface_blend_v2 needs --gallery (build one with build_gallery.py)")
        gal = load_gallery(args.gallery, args.column)
        apply = lambda e: arcface_blend_v2(e, args.key, gal, args.strength, args.n_mix)
    elif args.transform == "arcface_keymix_whitened_v3":
        if not args.basis:
            ap.error("arcface_keymix_whitened_v3 needs --basis (build one with build_basis.py)")
        bs = load_basis(args.basis, args.column)
        apply = lambda e: arcface_keymix_whitened_v3(e, args.key, bs, args.offset)
    else:
        apply = lambda e: arcface_keymix_v1(e, args.key, args.offset)

    inp = pathlib.Path(args.input)
    if inp.is_dir():
        outdir = pathlib.Path(args.output or (str(inp) + "_hashed"))
        outdir.mkdir(parents=True, exist_ok=True)
        for p in sorted(inp.rglob("*arcface.npy")):
            h = apply(np.load(p))
            op = outdir / p.name.replace("arcface", "arcface_hashed")
            np.save(op, h)
            print(f"  {p.name}  cos={cosine(np.load(p), h):+.3f}  -> {op}")
    else:
        e = np.load(inp)
        h = apply(e)
        op = args.output or str(inp).replace(".npy", "_hashed.npy")
        np.save(op, h)
        print(f"  {inp.name}: cos(original, hashed) = {cosine(e, h):+.4f}  ->  {op}")


if __name__ == "__main__":
    main()
