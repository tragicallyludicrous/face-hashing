"""
present.py — inspect & present the local pipeline's identity outputs.

(Named present.py, NOT inspect.py, on purpose: a top-level inspect.py shadows the stdlib `inspect`
module and breaks numpy/matplotlib imports.)

Trawls an output tree (default local/out) RECURSIVELY for identity vectors and renders separation
metrics. It matches files by *suffix*, so it handles both layouts:
  - run.py:      out/<stem>/{<stem>_arcface.npy, <stem>_shape.glb}
  - mica_local:  out/<stem>/{identity.npy, arcface.npy, <stem>.glb}
Needs only numpy + matplotlib (NOT torch/MICA):

    pip install matplotlib            # numpy already in the local/.venv

Pick which vector with --source:
    --source arcface   512-d ArcFace embedding (default; what run.py saves)  <- recognition 'ceiling'
    --source mica      300-d FLAME identity                                  <- the hash payload

For --source mica it reads identity.npy if present (mica_local), else RECOVERS the 300-d shape by
projecting the neutral _shape.glb onto the FLAME shape basis: the mesh is exactly
v_template + shapedirs[:, :, :300] @ beta, so the projection returns beta to ~1e-6. That path needs
the exported basis (viewer/flame/flame_basis.bin from tools/export_flame_basis.py) and trimesh to
read the .glb; both are loaded lazily, so the arcface path stays numpy-only.

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

SOURCES = {                                  # --source -> (label, [(filename suffix, kind), ...] in priority order)
    "arcface": ("ArcFace embedding", [("arcface.npy", "npy")]),
    "mica":    ("MICA identity",     [("identity.npy", "npy"), ("_shape.glb", "glb")]),
}


def person_of(stem):
    """'Person Context' or 'person-context' -> person is the first token (case-insensitive)."""
    return re.split(r"[ -]", stem.strip(), 1)[0].lower()


def _stem_of(path, suffix):
    """Photo stem from a path: '<stem>_arcface.npy' -> '<stem>', or a bare 'arcface.npy' -> parent dir."""
    base = os.path.basename(path)
    if base == suffix:                       # bare name -> the stem is the containing folder
        return os.path.basename(os.path.dirname(path))
    return base[: -len(suffix)].rstrip("_")  # '<stem>_arcface.npy' / '<stem>_shape.glb' -> '<stem>'


def find_vectors(root, candidates):
    """Recursively find vectors under root -> {stem: (path, kind)}. Earlier candidates win per stem."""
    allfiles = []
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()                          # deterministic traversal
        for f in sorted(files):
            allfiles.append((dirpath, f))
    found = {}
    for suffix, kind in candidates:          # priority order (e.g. identity.npy before _shape.glb)
        for dirpath, f in allfiles:
            if f.endswith(suffix):
                stem = _stem_of(os.path.join(dirpath, f), suffix)
                found.setdefault(stem, (os.path.join(dirpath, f), kind))
    return found


# ---- FLAME-basis projection: recover MICA's 300-d shape from a neutral _shape.glb ----------

_BASIS = None                                # (v_template_flat, pinv(shapedirs.T)) cached after first use


def _flame_basis():
    """Load the exported FLAME shape basis (numpy, no torch) and cache (v_template, pinv) for projection."""
    global _BASIS
    if _BASIS is None:
        bdir = os.path.join(ROOT, "viewer", "flame")
        binp, manp = os.path.join(bdir, "flame_basis.bin"), os.path.join(bdir, "flame_basis.json")
        if not os.path.exists(binp):
            raise SystemExit("recovering MICA shape from *_shape.glb needs the FLAME basis. Run:\n"
                             "  local/.venv/bin/python tools/export_flame_basis.py\n"
                             "(license-gated, git-ignored). Or use --source arcface.")
        man = json.load(open(manp)); nv, K = man["n_verts"], man["n_shape"]
        buf = np.fromfile(binp, dtype="<f4", count=3 * nv + K * 3 * nv)
        vt = buf[:3 * nv].astype(np.float64)                       # (3*nv,) vertex-major
        S = buf[3 * nv: 3 * nv + K * 3 * nv].reshape(K, 3 * nv).astype(np.float64)
        _BASIS = (vt, np.linalg.pinv(S.T))                         # pinv (K, 3*nv): one SVD up front
    return _BASIS


def _beta_from_glb(path):
    """Neutral _shape.glb -> 300-d MICA shape beta, by projecting verts onto the FLAME shape basis."""
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64) / 1000.0          # glb stores verts in mm -> meters
    vt, pinv = _flame_basis()
    d = V.reshape(-1) - vt                                         # vertex-major [x0,y0,z0,...] - v_template
    if d.size != pinv.shape[1]:
        raise SystemExit(f"{path}: {V.shape[0]} verts vs FLAME basis {pinv.shape[1] // 3} — basis/model mismatch.")
    return (pinv @ d).astype(np.float32)


def _load_vec(path, kind):
    return np.load(path).ravel() if kind == "npy" else _beta_from_glb(path)


def availability(root):
    """{source: count} found under root, for helpful 'nothing here / try the other source' messages."""
    return {src: len(find_vectors(root, cands)) for src, (_, cands) in SOURCES.items()}


def load_all(root, candidates):
    """-> (stems, persons, X[n,d]) for every vector found under root."""
    found = find_vectors(root, candidates)
    stems = sorted(found)
    X = np.stack([_load_vec(*found[s]) for s in stems]) if stems else np.empty((0, 0))
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

def single(root, stem, candidates, label, source):
    found = find_vectors(root, candidates)
    if stem not in found:
        raise SystemExit(f"no {label} vector for '{stem}' under {root}. "
                         f"Available: {', '.join(sorted(found)) or '(none)'}")
    path, kind = found[stem]
    v = _load_vec(path, kind)
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
    vec_dir = os.path.dirname(path)
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


def compare(root, candidates, label, source):
    stems, persons, X = load_all(root, candidates)
    if len(stems) < 2:
        avail = availability(root)
        hint = f" Try --source {max(avail, key=avail.get)}." if max(avail.values(), default=0) >= 2 else ""
        raise SystemExit(f"need >=2 {label} vectors under {root} to compare; found "
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
    # one tick per identity, centered on its block — per-photo labels are an unreadable comb at this count
    edges = [k for k in range(1, len(persons)) if persons[k] != persons[k - 1]]
    bounds = [0] + edges + [len(persons)]
    centers = [(bounds[i] + bounds[i + 1] - 1) / 2 for i in range(len(bounds) - 1)]
    names = [persons[bounds[i]].capitalize() for i in range(len(bounds) - 1)]
    plt.xticks(centers, names, rotation=45, ha="right", fontsize=9)
    plt.yticks(centers, names, fontsize=9)
    for b in edges:
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
    label, candidates = SOURCES[a.source]
    if a.compare:
        compare(root, candidates, label, a.source)
    elif a.stem:
        single(root, a.stem, candidates, label, a.source)
    else:
        avail = availability(root)
        ap.error(f"give a stem or --compare. Found under {a.dir}: "
                 + ", ".join(f"{s}={n}" for s, n in avail.items()))
