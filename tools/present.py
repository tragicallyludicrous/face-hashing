"""
present.py — inspect & present MICA identity / ArcFace outputs.

(Named present.py, NOT inspect.py, on purpose: a top-level inspect.py shadows the stdlib `inspect`
module and breaks numpy/matplotlib imports.)

Operates on the local results in  local/out/<stem>/{identity.npy, arcface.npy, <stem>.glb}
(written by local/mica_local.py). Needs only numpy + matplotlib (NOT torch/MICA):

    pip install matplotlib            # numpy already in the local/.venv

Pick which vector with --source:
    --source mica      300-d FLAME identity (default)   <- the hash payload
    --source arcface   512-d ArcFace embedding          <- the recognition 'ceiling' MICA consumes

Single photo — JSON of the vector, a "fingerprint" image, and copy the mesh to the viewer:

    python tools/present.py <stem> [--source mica|arcface]
      -> tools/figures/<stem>_<source>_identity.json
         tools/figures/<stem>_<source>_fingerprint.png
         copies local/out/<stem>/<stem>.glb -> viewer/models/

Separation view — the presentable part. Groups photos by person from the "person-context"
filename and renders the relationships (raw per-coefficient overlays are noisy; THESE separate
people):

    python tools/present.py --compare [--source mica|arcface]
      -> tools/figures/distance_heatmap_<source>.png   (same-person = dark blocks)
         tools/figures/pca_scatter_<source>.png        (each person a cluster)
         prints intra/inter cosine distance + verification AUC

Run it twice (--source mica, then --source arcface) to compare the FLAME identity against the
ArcFace ceiling on the same photos.
"""
import argparse
import itertools
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: just write PNGs
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "local", "out")            # MICA results
VIEWER_MODELS = os.path.join(ROOT, "viewer", "models")
FIG = os.path.join(ROOT, "tools", "figures")

SOURCES = {                                          # --source -> (filename, label)
    "mica":    ("identity.npy", "MICA identity"),
    "arcface": ("arcface.npy",  "ArcFace embedding"),
}


def person_of(stem):
    """Filenames are 'person-context' -> person is everything before the first hyphen."""
    return stem.split("-", 1)[0]


def list_stems(fname):
    if not os.path.isdir(OUT):
        return []
    return sorted(d for d in os.listdir(OUT) if os.path.exists(os.path.join(OUT, d, fname)))


def load_all(fname):
    """-> (stems, persons, X[n,d])."""
    stems = list_stems(fname)
    X = np.stack([np.load(os.path.join(OUT, s, fname)).ravel() for s in stems])
    return stems, [person_of(s) for s in stems], X


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cosine_dist(X):
    Xn = _unit(X)
    return 1.0 - Xn @ Xn.T


def pca_2d(X):
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _to_grid(v):
    """Reshape a 1-D vector into a near-square grid (zero-padded) for the fingerprint tile."""
    cols = int(np.ceil(np.sqrt(v.size)))
    rows = int(np.ceil(v.size / cols))
    g = np.zeros(rows * cols, dtype=float)
    g[: v.size] = v
    return g.reshape(rows, cols)


# ---- single-photo inspection ---------------------------------------------------------------

def single(stem, fname, label, source):
    p = os.path.join(OUT, stem, fname)
    if not os.path.exists(p):
        raise SystemExit(f"no {fname} for '{stem}'. Available: {', '.join(list_stems(fname)) or '(none)'}")
    v = np.load(p).ravel()
    os.makedirs(FIG, exist_ok=True)

    # 1) JSON of the raw vector
    jpath = os.path.join(FIG, f"{stem}_{source}_identity.json")
    json.dump({"stem": stem, "person": person_of(stem), "source": source, "dim": int(v.size),
               "vector": np.round(v, 4).tolist()}, open(jpath, "w"), indent=2)
    print(f"{label}[{v.size}] for {stem}: [{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f} … ]  -> {jpath}")

    # 2) "fingerprint": vector reshaped to a near-square heat-tile; each person reads distinctly
    grid = _to_grid(v)
    lim = np.abs(grid).max()
    plt.figure(figsize=(5, 4))
    plt.imshow(grid, cmap="coolwarm", vmin=-lim, vmax=lim)
    plt.title(f"{label} fingerprint — {stem}")
    plt.axis("off"); plt.colorbar(fraction=0.046)
    fp = os.path.join(FIG, f"{stem}_{source}_fingerprint.png")
    plt.tight_layout(); plt.savefig(fp, dpi=150); plt.close()
    print("fingerprint ->", fp)

    # 3) copy the mesh into the viewer (drag-rotate at http://localhost:8080)
    glb = os.path.join(OUT, stem, f"{stem}.glb")
    if os.path.exists(glb):
        os.makedirs(VIEWER_MODELS, exist_ok=True)
        shutil.copy(glb, VIEWER_MODELS)
        print(f"mesh   -> {os.path.join('viewer','models', stem + '.glb')}  (serve: cd viewer && python3 -m http.server 8080)")
    else:
        print(f"(no {stem}.glb to copy)")


# ---- multi-photo separation view -----------------------------------------------------------

def _auc(intra, inter):
    if not len(intra) or not len(inter):
        return float("nan")
    return sum(float((a < inter).sum() + 0.5 * (a == inter).sum()) for a in intra) / (len(intra) * len(inter))


def compare(fname, label, source):
    stems, persons, X = load_all(fname)
    if len(stems) < 2:
        raise SystemExit(f"need >=2 photos with {fname} in local/out/ to compare")
    os.makedirs(FIG, exist_ok=True)

    # order by person so same-person photos are adjacent (dark blocks on the diagonal)
    order = sorted(range(len(stems)), key=lambda i: (persons[i], stems[i]))
    stems = [stems[i] for i in order]; persons = [persons[i] for i in order]; X = X[order]
    D = cosine_dist(X)

    intra, inter = [], []
    for i, j in itertools.combinations(range(len(X)), 2):
        (intra if persons[i] == persons[j] else inter).append(D[i, j])
    intra, inter = np.array(intra), np.array(inter)
    if len(intra) and len(inter):
        print(f"[{label}] cosine dist  intra={intra.mean():.3f}  inter={inter.mean():.3f}  "
              f"sep={inter.mean()/(intra.mean()+1e-9):.2f}x  AUC={_auc(intra, inter):.3f}")
    else:
        print(f"[{label}] (need >=2 photos of one person AND >=2 people for intra/inter stats)")

    # 1) distance heatmap
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(D, cmap="magma")
    plt.colorbar(label="cosine distance (dark = same identity)")
    plt.xticks(range(len(stems)), stems, rotation=90, fontsize=7)
    plt.yticks(range(len(stems)), stems, fontsize=7)
    for b in [k for k in range(1, len(persons)) if persons[k] != persons[k - 1]]:
        plt.axhline(b - 0.5, color="w", lw=1); plt.axvline(b - 0.5, color="w", lw=1)
    plt.title(f"{label} distance — same person clusters into dark blocks")
    hm = os.path.join(FIG, f"distance_heatmap_{source}.png")
    plt.tight_layout(); plt.savefig(hm, dpi=150); plt.close()
    print("heatmap ->", hm)

    # 2) PCA scatter (each person a cluster)
    P = pca_2d(X)
    uniq = sorted(set(persons))
    cmap = plt.get_cmap("tab10")
    color = {p: cmap(k % 10) for k, p in enumerate(uniq)}
    plt.figure(figsize=(6.5, 5.5))
    for p in uniq:
        idx = [k for k, q in enumerate(persons) if q == p]
        plt.scatter(P[idx, 0], P[idx, 1], color=color[p], label=p, s=80, edgecolor="k", linewidth=0.5)
    for k, s in enumerate(stems):
        plt.annotate(s, (P[k, 0], P[k, 1]), fontsize=6, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    plt.legend(title="person"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"{label} space (PCA to 2D)")
    sc = os.path.join(FIG, f"pca_scatter_{source}.png")
    plt.tight_layout(); plt.savefig(sc, dpi=150); plt.close()
    print("scatter ->", sc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect/present MICA identity or ArcFace outputs (local/out/).")
    ap.add_argument("stem", nargs="?", help="a photo stem under local/out/ (single-photo inspection)")
    ap.add_argument("--compare", action="store_true", help="separation view across all photos")
    ap.add_argument("--source", choices=list(SOURCES), default="mica",
                    help="which vector: mica (300-d identity, default) or arcface (512-d embedding)")
    a = ap.parse_args()
    fname, label = SOURCES[a.source]
    if a.compare:
        compare(fname, label, a.source)
    elif a.stem:
        single(a.stem, fname, label, a.source)
    else:
        ap.error(f"give a stem or --compare. Available: {', '.join(list_stems(fname)) or '(none in local/out)'}")
