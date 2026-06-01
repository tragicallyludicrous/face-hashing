"""
present.py — inspect & present MICA identity / FLAME outputs.

(Named present.py, NOT inspect.py, on purpose: a top-level inspect.py shadows the stdlib `inspect`
module and breaks numpy/matplotlib imports.)

Operates on the local MICA results in  local/out/<stem>/{identity.npy, <stem>.glb}.
Needs only numpy + matplotlib (NOT torch/MICA), so run it with any Python that has those:

    pip install matplotlib            # numpy already in the local/.venv

Single photo — JSON of the vector, an "identity fingerprint" image, and copy the mesh to the viewer:

    python tools/present.py <stem>
      -> prints + writes tools/figures/<stem>_identity.json
         writes  tools/figures/<stem>_fingerprint.png   (300-d reshaped to a heat-tile)
         copies  local/out/<stem>/<stem>.glb -> viewer/models/   (then serve the viewer)

Separation view — the presentable part (raw 300-d overlays are noisy; THIS is what differentiates
people). Groups photos by person from the "person-context" filename and renders the relationships:

    python tools/present.py --compare
      -> tools/figures/distance_heatmap.png   (same-person = dark blocks)
         tools/figures/pca_scatter.png        (each person a cluster)
         prints intra/inter cosine distance + verification AUC
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


def person_of(stem):
    """Filenames are 'person-context' -> person is everything before the first hyphen."""
    return stem.split("-", 1)[0]


def list_stems():
    if not os.path.isdir(OUT):
        return []
    return sorted(d for d in os.listdir(OUT)
                  if os.path.exists(os.path.join(OUT, d, "identity.npy")))


def load_all():
    """-> (stems, persons, X[n,300])."""
    stems = list_stems()
    X = np.stack([np.load(os.path.join(OUT, s, "identity.npy")).ravel() for s in stems])
    persons = [person_of(s) for s in stems]
    return stems, persons, X


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cosine_dist(X):
    Xn = _unit(X)
    return 1.0 - Xn @ Xn.T


def pca_2d(X):
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# ---- single-photo inspection ---------------------------------------------------------------

def single(stem):
    p = os.path.join(OUT, stem, "identity.npy")
    if not os.path.exists(p):
        raise SystemExit(f"no identity for '{stem}'. Available: {', '.join(list_stems()) or '(none)'}")
    v = np.load(p).ravel()
    os.makedirs(FIG, exist_ok=True)

    # 1) JSON of the raw vector (the "facial features as numbers" prop)
    jpath = os.path.join(FIG, f"{stem}_identity.json")
    json.dump({"stem": stem, "person": person_of(stem), "dim": int(v.size),
               "identity": np.round(v, 4).tolist()}, open(jpath, "w"), indent=2)
    print(f"identity[{v.size}] for {stem}: [{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f} … ]  -> {jpath}")

    # 2) "fingerprint": 300-d reshaped to a tile grid; each person reads as a distinct pattern
    side = 15
    grid = v[: side * (v.size // side)].reshape(side, -1)
    lim = np.abs(grid).max()
    plt.figure(figsize=(5, 4))
    plt.imshow(grid, cmap="coolwarm", vmin=-lim, vmax=lim)
    plt.title(f"MICA identity fingerprint — {stem}")
    plt.axis("off"); plt.colorbar(fraction=0.046)
    fp = os.path.join(FIG, f"{stem}_fingerprint.png")
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


def compare():
    stems, persons, X = load_all()
    if len(stems) < 2:
        raise SystemExit("need >=2 photos in local/out/ to compare")
    os.makedirs(FIG, exist_ok=True)

    # order by person so same-person photos are adjacent (dark blocks on the diagonal)
    order = sorted(range(len(stems)), key=lambda i: (persons[i], stems[i]))
    stems = [stems[i] for i in order]; persons = [persons[i] for i in order]; X = X[order]
    D = cosine_dist(X)

    # printed stats (ties the picture to a number)
    intra, inter = [], []
    for i, j in itertools.combinations(range(len(X)), 2):
        (intra if persons[i] == persons[j] else inter).append(D[i, j])
    intra, inter = np.array(intra), np.array(inter)
    if len(intra) and len(inter):
        print(f"cosine dist  intra={intra.mean():.3f}  inter={inter.mean():.3f}  "
              f"sep={inter.mean()/(intra.mean()+1e-9):.2f}x  AUC={_auc(intra, inter):.3f}")
    else:
        print("(need >=2 photos of one person AND >=2 people for intra/inter stats)")

    # 1) distance heatmap
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(D, cmap="magma")
    plt.colorbar(label="cosine distance (dark = same identity)")
    plt.xticks(range(len(stems)), stems, rotation=90, fontsize=7)
    plt.yticks(range(len(stems)), stems, fontsize=7)
    # white lines at person-group boundaries
    bounds = [k for k in range(1, len(persons)) if persons[k] != persons[k - 1]]
    for b in bounds:
        plt.axhline(b - 0.5, color="w", lw=1); plt.axvline(b - 0.5, color="w", lw=1)
    plt.title("Identity distance — same person clusters into dark blocks")
    hm = os.path.join(FIG, "distance_heatmap.png")
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
        plt.annotate(s, (P[k, 0], P[k, 1]), fontsize=6, alpha=0.7,
                     xytext=(4, 2), textcoords="offset points")
    plt.legend(title="person"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("MICA identity space (PCA to 2D)")
    sc = os.path.join(FIG, "pca_scatter.png")
    plt.tight_layout(); plt.savefig(sc, dpi=150); plt.close()
    print("scatter ->", sc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect/present MICA identity outputs (local/out/).")
    ap.add_argument("stem", nargs="?", help="a photo stem under local/out/ (single-photo inspection)")
    ap.add_argument("--compare", action="store_true", help="separation view across all photos")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.stem:
        single(a.stem)
    else:
        ap.error(f"give a stem or --compare. Available: {', '.join(list_stems()) or '(none in local/out)'}")
