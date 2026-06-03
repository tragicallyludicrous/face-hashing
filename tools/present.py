"""
present.py — inspect & present the local pipeline's identity outputs.

(Named present.py, NOT inspect.py, on purpose: a top-level inspect.py shadows the stdlib `inspect`
module and breaks numpy/matplotlib imports.)

Trawls an output tree (default local/out) RECURSIVELY for identity vectors and renders separation
metrics. It matches files by *suffix*, so it handles both layouts:
  - run.py:      out/<stem>/<stem>_arcface.npy
  - mica_local:  out/<stem>/{identity.npy, arcface.npy}
Needs only numpy + matplotlib (NOT torch/MICA):

    pip install matplotlib            # numpy already in the local/.venv

Pick which vector with --source:
    --source arcface   512-d ArcFace embedding (default; what run.py saves)  <- recognition 'ceiling'
    --source mica      300-d FLAME identity (identity.npy)                    <- the hash payload

Separation view — the presentable part. Groups photos by person from the 'Person Context' /
'person-context' filename and renders the relationships (raw per-coefficient overlays are noisy;
THESE separate people):

    python tools/present.py --compare [--source arcface|mica] [--dir local/out]
      -> tools/figures/distance_heatmap_<source>.png   (same person = dark blocks)
         tools/figures/pca_scatter_<source>.png        (each person a cluster)
         prints intra/inter cosine distance + verification AUC

Single photo — JSON of the vector, a 'fingerprint' image, and copy its mesh(es) to the viewer:

    python tools/present.py "<stem>" [--source arcface|mica] [--dir local/out]
"""
import argparse
import itertools
import json
import os
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: just write PNGs
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "local", "out")
VIEWER_MODELS = os.path.join(ROOT, "viewer", "models")
FIG = os.path.join(ROOT, "tools", "figures")

SOURCES = {                                  # --source -> (filename suffix, label)
    "arcface": ("arcface.npy",  "ArcFace embedding"),
    "mica":    ("identity.npy", "MICA identity"),
}


def person_of(stem):
    """'Person Context' or 'person-context' -> person is the first token (case-insensitive)."""
    return re.split(r"[ -]", stem.strip(), 1)[0].lower()


def _stem_of(path, suffix):
    """Photo stem from a vector path: '<stem>_arcface.npy' -> '<stem>', or legacy 'arcface.npy' -> parent dir."""
    base = os.path.basename(path)
    if base == suffix:                       # bare name -> the stem is the containing folder
        return os.path.basename(os.path.dirname(path))
    return base[: -len(suffix)].rstrip("_")  # '<stem>_arcface.npy' -> '<stem>'


def find_vectors(root, suffix):
    """Recursively find files ending in <suffix> under root -> {stem: path} (first hit per stem)."""
    found = {}
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()                          # deterministic traversal
        for f in sorted(files):
            if f.endswith(suffix):
                found.setdefault(_stem_of(os.path.join(dirpath, f), suffix), os.path.join(dirpath, f))
    return found


def availability(root):
    """{source: count} found under root, for helpful 'nothing here / try the other source' messages."""
    return {src: len(find_vectors(root, suf)) for src, (suf, _) in SOURCES.items()}


def load_all(root, suffix):
    """-> (stems, persons, X[n,d]) for every vector found under root."""
    found = find_vectors(root, suffix)
    stems = sorted(found)
    X = np.stack([np.load(found[s]).ravel() for s in stems]) if stems else np.empty((0, 0))
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

def single(root, stem, suffix, label, source):
    found = find_vectors(root, suffix)
    if stem not in found:
        raise SystemExit(f"no '{suffix}' for '{stem}' under {root}. "
                         f"Available: {', '.join(sorted(found)) or '(none)'}")
    v = np.load(found[stem]).ravel()
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

    # 3) copy the mesh(es) sitting beside the vector into the viewer (drag-rotate at :8080)
    vec_dir = os.path.dirname(found[stem])
    glbs = sorted(g for g in os.listdir(vec_dir) if g.startswith(stem) and g.endswith(".glb"))
    if glbs:
        os.makedirs(VIEWER_MODELS, exist_ok=True)
        for g in glbs:
            shutil.copy(os.path.join(vec_dir, g), VIEWER_MODELS)
        print("mesh(es) ->", ", ".join(os.path.join("viewer", "models", g) for g in glbs),
              " (serve: cd viewer && python3 -m http.server 8080)")
    else:
        print(f"(no {stem}*.glb beside the vector to copy)")


# ---- multi-photo separation view -----------------------------------------------------------

def _auc(intra, inter):
    if not len(intra) or not len(inter):
        return float("nan")
    return sum(float((a < inter).sum() + 0.5 * (a == inter).sum()) for a in intra) / (len(intra) * len(inter))


def compare(root, suffix, label, source):
    stems, persons, X = load_all(root, suffix)
    if len(stems) < 2:
        avail = availability(root)
        hint = f" Try --source {max(avail, key=avail.get)}." if max(avail.values(), default=0) >= 2 else ""
        raise SystemExit(f"need >=2 '{suffix}' vectors under {root} to compare; found "
                         + ", ".join(f"{s}={n}" for s, n in avail.items()) + "." + hint)
    os.makedirs(FIG, exist_ok=True)

    # order by person so same-person photos are adjacent (dark blocks on the diagonal)
    order = sorted(range(len(stems)), key=lambda i: (persons[i], stems[i]))
    stems = [stems[i] for i in order]; persons = [persons[i] for i in order]; X = X[order]
    D = cosine_dist(X)

    intra, inter = [], []
    for i, j in itertools.combinations(range(len(X)), 2):
        (intra if persons[i] == persons[j] else inter).append(D[i, j])
    intra, inter = np.array(intra), np.array(inter)
    n_people = len(set(persons))
    if len(intra) and len(inter):
        print(f"[{label}] {len(stems)} photos, {n_people} people  |  cosine dist  "
              f"intra={intra.mean():.3f}  inter={inter.mean():.3f}  "
              f"sep={inter.mean()/(intra.mean()+1e-9):.2f}x  AUC={_auc(intra, inter):.3f}")
    else:
        print(f"[{label}] {len(stems)} photos, {n_people} people "
              f"(need >=2 photos of one person AND >=2 people for intra/inter stats)")

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
    ap = argparse.ArgumentParser(description="Inspect/present the local pipeline's identity outputs (trawls --dir).")
    ap.add_argument("stem", nargs="?", help="a photo stem to inspect (single-photo mode)")
    ap.add_argument("--compare", action="store_true", help="separation view across all photos found")
    ap.add_argument("--source", choices=list(SOURCES), default="arcface",
                    help="which vector: arcface (512-d, default; what run.py saves) or mica (300-d identity.npy)")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="output tree to trawl recursively (default local/out)")
    a = ap.parse_args()
    root = os.path.abspath(a.dir)
    suffix, label = SOURCES[a.source]
    if a.compare:
        compare(root, suffix, label, a.source)
    elif a.stem:
        single(root, a.stem, suffix, label, a.source)
    else:
        avail = availability(root)
        ap.error(f"give a stem or --compare. Found under {a.dir}: "
                 + ", ".join(f"{s}={n}" for s, n in avail.items()))
