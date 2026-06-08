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


# registry hook (Stage-2 plan: hot-swappable strategies)
TRANSFORMS = {"arcface_keymix_v1": arcface_keymix_v1}


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

    inp = pathlib.Path(args.input)
    if inp.is_dir():
        outdir = pathlib.Path(args.output or (str(inp) + "_hashed"))
        outdir.mkdir(parents=True, exist_ok=True)
        for p in sorted(inp.rglob("*arcface.npy")):
            h = arcface_keymix_v1(np.load(p), args.key, args.offset)
            op = outdir / p.name.replace("arcface", "arcface_hashed")
            np.save(op, h)
            print(f"  {p.name}  cos={cosine(np.load(p), h):+.3f}  -> {op}")
    else:
        e = np.load(inp)
        h = arcface_keymix_v1(e, args.key, args.offset)
        op = args.output or str(inp).replace(".npy", "_hashed.npy")
        np.save(op, h)
        print(f"  {inp.name}: cos(original, hashed) = {cosine(e, h):+.4f}  ->  {op}")


if __name__ == "__main__":
    main()
