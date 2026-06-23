"""build_basis.py — fit the whitening basis for `arcface_keymix_whitened_v3`.

Input: an embeddings `.npz` from `build_gallery.py`, which you run over a folder of **synthetic**
faces — "people who don't exist" (StyleGAN / SFHQ / thispersondoesnotexist). No real identity is
ever involved: we only measure the *shape* of the human-face region of ArcFace space.

For each column present ('antelope' = InstantID texture, 'mica' = depth geometry) it fits the
identity subspace and writes `basis.npz` with `<col>_mu`, `<col>_V`, `<col>_sigma`:

    mu    : (D,)   mean embedding
    V     : (k,D)  top-k PCA axes — the manifold's dominant identity directions
    sigma : (k,)   per-axis std, for whitening

These are AGGREGATE statistics; no faces are retained, no individual is targeted. `v3` permutes in
this whitened space (where a permutation preserves the distribution), so its output is a *derived
synthetic* identity that stays on the manifold.

    python build_gallery.py -i synthetic_faces/ -o corpus.npz       # extract embeddings (needs MICA)
    python build_basis.py  -i corpus.npz -o basis.npz --var 0.95    # fit the basis (pure numpy)
"""
import argparse

import numpy as np


def fit(E, var=0.95, kmax=0):
    """E: (N,D) -> (mu (D,), V (k,D), sigma (k,)). Keep enough PCA axes for `var` of the variance."""
    E = np.asarray(E, np.float64)
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)   # principal axes = rows of Vt
    ev = (S ** 2) / max(len(E) - 1, 1)                       # per-axis variance (eigenvalues)
    cum = np.cumsum(ev) / np.sum(ev)
    k = int(np.searchsorted(cum, var) + 1)
    if kmax:
        k = min(k, kmax)
    k = max(1, min(k, len(S)))
    return mu, Vt[:k], np.sqrt(ev[:k])


def main():
    ap = argparse.ArgumentParser(description="Fit the whitening basis for arcface_keymix_whitened_v3.")
    ap.add_argument("-i", "--in", dest="inp", required=True, help="embeddings .npz from build_gallery.py")
    ap.add_argument("-o", "--out", default="basis.npz")
    ap.add_argument("--var", type=float, default=0.95, help="PCA variance fraction to keep (0..1)")
    ap.add_argument("--kmax", type=int, default=0, help="cap # components (0 = no cap)")
    a = ap.parse_args()

    z = np.load(a.inp)
    out = {}
    for col in ("antelope", "mica"):
        if col not in z:
            continue
        E = z[col]
        if E.ndim != 2 or len(E) < 8:
            print(f"  skip {col}: need (N>=8, D), got {getattr(E, 'shape', None)}"); continue
        if len(E) <= E.shape[1]:
            print(f"  ! {col}: {len(E)} samples for dim {E.shape[1]} — rank-limited; use more faces")
        mu, V, sigma = fit(E, a.var, a.kmax)
        out[col + "_mu"] = mu.astype(np.float32)
        out[col + "_V"] = V.astype(np.float32)
        out[col + "_sigma"] = sigma.astype(np.float32)
        print(f"  {col}: {len(E)} embeddings -> {V.shape[0]} axes (>= {a.var:.0%} var), D={E.shape[1]}")
    if not out:
        raise SystemExit("no 'antelope'/'mica' columns in the input .npz")
    np.savez(a.out, **out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
