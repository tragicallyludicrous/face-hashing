"""
pipeline.py — Face Hashing, Stage 1 (in-kernel DECA, no renderer).

Runs in Google Colab. Keeps the notebook thin:

    !git clone https://github.com/tragicallyludicrous/face-hashing.git
    import sys; sys.path.insert(0, "face-hashing")
    import pipeline

    pipeline.bootstrap()                  # clone DECA, install, patch, cache weights (idempotent)
    deca, faces = pipeline.load_deca()    # renderer-free DECA + FLAME topology
    pipeline.reconstruct(deca, faces)     # Drive Input/ photos -> Output/<name>/{.glb,_params.npz}
    pipeline.tweak(deca, faces, "<name>") # Skyrim-slider: mutate identity, re-export <name>_tweaked.glb

Why this shape (see CONTEXT.md "Key decision 2026-05-29"):
- DECA's custom CUDA rasterizer is OPTIONAL — our deliverables (param dict + mesh) come from
  encode() (a ResNet) and the FLAME decoder, neither of which renders. We neuter the renderer
  (`DECA._setup_renderer`) so the gcc-7 / PyTorch3D build never happens.
- Running in-kernel means the chumpy fix is a simple in-process monkeypatch (no on-disk shim),
  because there is no fresh `!python` subprocess to lose it.
- Google Drive is the cross-restart cache: the 434 MB DECA weights are gdown'd ONCE into
  cache/ and copied locally each session; nothing big re-downloads.

decalib / chumpy are imported LAZILY (after bootstrap applies the chumpy patch), never at
module top level — so this file is safe to import anywhere (e.g. for editing/linting).

Two DECA-API details (the FLAME faces attribute and the flame() kwargs) vary by revision;
both are verified on the current DECA build, with fallbacks inline.
"""

import os
import sys
import subprocess

# --- DECA source + weights -------------------------------------------------
DECA_DIR = "/content/DECA"
DECA_REPO = "https://github.com/yfeng95/DECA.git"
DECA_GDRIVE_ID = "1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje"  # deca_model.tar (~434 MB)

# --- Google Drive layout (the handoff + cross-restart cache) ----------------
DRIVE = "/content/drive/MyDrive/Face-Hashing"
CACHE = f"{DRIVE}/cache"
IN_DIR = f"{DRIVE}/Input"
OUT_DIR = f"{DRIVE}/Output"
FLAME_PKL = f"{DRIVE}/FLAME/FLAME2020/generic_model.pkl"

PIP_PKGS = [
    "chumpy", "yacs==0.1.8", "face-alignment", "ninja", "kornia==0.6.12",
    "scikit-image", "opencv-python", "PyYAML", "trimesh", "gdown",
]


def _sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def _patch_chumpy_inproc():
    """Restore inspect.getargspec + numpy aliases BEFORE chumpy is imported.

    chumpy is unmaintained: it calls inspect.getargspec (removed in 3.11+) and does
    `from numpy import bool, int, ...` (removed in numpy 2.x). An in-kernel monkeypatch is
    sufficient here because we never spawn a fresh `python` subprocess that would miss it.
    """
    import inspect
    import numpy as np
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for alias, real in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("str", str), ("unicode", str)]:
        if not hasattr(np, alias):
            setattr(np, alias, real)


def _mount_drive():
    if os.path.isdir(DRIVE):
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("Not running in Colab; skipping Drive mount.")


def _patch_detector():
    """face-alignment renamed LandmarksType._2D -> TWO_D; DECA's detector still uses the old name."""
    det = f"{DECA_DIR}/decalib/datasets/detectors.py"
    src = open(det).read()
    if "LandmarksType._2D" in src:
        open(det, "w").write(src.replace("LandmarksType._2D", "LandmarksType.TWO_D"))


def _fetch_weights():
    """gdown the DECA weights ONCE into the Drive cache, then copy locally each session.

    Copying Drive -> local is a fast internal copy (no internet re-download, no gdown
    throttling) and loading from local disk is reliable.
    """
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(f"{DECA_DIR}/data", exist_ok=True)
    deca_tar = f"{CACHE}/deca_model.tar"
    if not os.path.exists(deca_tar):  # first session ever; skipped on every restart after
        _sh(f'gdown {DECA_GDRIVE_ID} -O "{deca_tar}"')
    _sh(f'cp -n "{deca_tar}" "{DECA_DIR}/data/deca_model.tar"')  # -n: don't re-copy within a session
    if os.path.exists(FLAME_PKL):
        _sh(f'cp -n "{FLAME_PKL}" "{DECA_DIR}/data/generic_model.pkl"')
    else:
        print(f"WARNING: FLAME model not found at {FLAME_PKL} — place generic_model.pkl there.")


def bootstrap(install=True):
    """Idempotent cold-start setup. Safe to re-run; does the minimum each time.

    Mount Drive -> clone DECA (if missing) -> pip install deps -> patch detector + chumpy ->
    fetch weights (gdown once -> Drive cache -> local) -> put DECA on sys.path.
    Call this once at the top of a fresh Colab session, before load_deca().
    """
    _mount_drive()
    if not os.path.isdir(DECA_DIR):
        _sh(f"git clone {DECA_REPO} {DECA_DIR}")
    if install:
        _sh(f"{sys.executable} -m pip install -q " + " ".join(PIP_PKGS))
    _patch_detector()
    _fetch_weights()
    _patch_chumpy_inproc()  # must precede any decalib/chumpy import
    if DECA_DIR not in sys.path:
        sys.path.insert(0, DECA_DIR)
    print("bootstrap complete.")


def load_deca(device="cuda", use_tex=False):
    """Construct DECA WITHOUT its CUDA rasterizer and return (deca, faces).

    `DECA._setup_renderer` is the only thing that triggers the gcc-7 / PyTorch3D build; we
    replace it with a no-op. The mesh's faces (topology) come from the FLAME model, not the
    renderer. Must be called after bootstrap() (which applies the chumpy patch + sys.path).
    """
    os.chdir(DECA_DIR)  # DECA's config resolves data/ paths relative to its own dir
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    DECA._setup_renderer = lambda self, model_cfg: None  # <- skips the entire CUDA rasterizer build
    deca_cfg.model.use_tex = use_tex
    deca = DECA(config=deca_cfg, device=device)

    try:
        faces = deca.flame.faces_tensor.detach().cpu().numpy()
    except AttributeError:  # some DECA revisions expose the topology differently
        import trimesh
        faces = trimesh.load(deca_cfg.model.topology_path, process=False).faces  # fallback: template obj
    return deca, faces


def _decode_verts(deca, shape, exp, pose):
    """FLAME params -> mesh vertices, calling the decoder directly (no renderer)."""
    import torch
    with torch.no_grad():
        verts, _, _ = deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
    return verts


def _export_glb(verts, faces, path, color=(180, 180, 200, 255)):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(), faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=list(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def _prepare_inputs(in_dir, max_side=1024, tmp="/content/_resized"):
    """Downscale (<=max_side) + EXIF-rotate every image into tmp, return the list of temp paths.

    The FAN detector runs on the FULL image and OOMs a free-tier T4 on large phone photos, while
    DECA only needs ~224 px around the face. Passing an explicit list also sidesteps DECA's
    TestData directory glob, which misses .jpeg (it only globs *.jpg/*.png/*.bmp).
    """
    from PIL import Image, ImageOps
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for f in sorted(os.listdir(in_dir)):
        if os.path.splitext(f)[1].lower() not in exts:
            continue
        im = ImageOps.exif_transpose(Image.open(os.path.join(in_dir, f)).convert("RGB"))
        im.thumbnail((max_side, max_side))
        p = os.path.join(tmp, os.path.splitext(f)[0] + ".jpg")
        im.save(p, quality=95)
        paths.append(p)
    return paths


def reconstruct(deca, faces, in_dir=IN_DIR, out_dir=OUT_DIR, detector="fan", max_side=1024):
    """Every photo in in_dir -> out_dir/<name>/{<name>.glb, <name>_params.npz}. Returns the names."""
    import gc
    import torch
    import numpy as np
    from decalib.datasets import datasets

    paths = _prepare_inputs(in_dir, max_side=max_side)
    data = datasets.TestData(paths, iscrop=True, face_detector=detector)  # FAN detector crops to 224
    names = []
    for i in range(len(data)):
        with torch.no_grad():
            d = data[i]
            name = d["imagename"]
            images = d["image"].to(deca.device)[None, ...]
            codedict = deca.encode(images)
            verts = _decode_verts(deca, codedict["shape"], codedict["exp"], codedict["pose"])
        _export_glb(verts, faces, f"{out_dir}/{name}/{name}.glb")
        params = {k: v.detach().cpu().numpy()
                  for k, v in codedict.items() if torch.is_tensor(v) and k != "images"}
        np.savez(f"{out_dir}/{name}/{name}_params.npz", **params)
        del images, codedict, verts  # free GPU memory between images (T4 headroom is tight)
        gc.collect()
        torch.cuda.empty_cache()
        print("ok", name, "| shape", params["shape"].shape, "exp", params["exp"].shape)
        names.append(name)
    return names


def default_mutation(shape):
    """Placeholder identity tweak. Stage 2 replaces this with a deterministic, key-seeded transform."""
    shape = shape.clone()
    shape[0, 0] *= -1.0  # flip the biggest shape PC (~face width)
    shape[0, 1] *= 1.5   # exaggerate the next mode
    return shape


def tweak(deca, faces, name, mutate=default_mutation, out_dir=OUT_DIR, color=(200, 180, 180, 255)):
    """Load saved params for <name>, mutate IDENTITY only (keep exp/pose), export <name>_tweaked.glb."""
    import torch
    import numpy as np
    p = np.load(f"{out_dir}/{name}/{name}_params.npz")
    shape = mutate(torch.tensor(p["shape"]).to(deca.device))
    exp = torch.tensor(p["exp"]).to(deca.device)
    pose = torch.tensor(p["pose"]).to(deca.device)
    verts = _decode_verts(deca, shape, exp, pose)
    _export_glb(verts, faces, f"{out_dir}/{name}/{name}_tweaked.glb", color=color)
    print("exported", f"{name}_tweaked.glb")


def export_neutral(deca, faces, name, out_dir=OUT_DIR, keep_expression=False,
                   color=(180, 180, 200, 255)):
    """Re-export <name> with global head pose zeroed (and expression too, unless keep_expression).

    DECA bakes the source photo's head rotation into the mesh via the pose param, so meshes from
    different photos point different ways and never line up in the viewer. Zeroing pose puts every
    identity in one canonical, frontal orientation -> switching models compares geometry, not pose.
    Writes <name>_neutral.glb alongside the others.
    """
    import torch
    import numpy as np
    p = np.load(f"{out_dir}/{name}/{name}_params.npz")
    shape = torch.tensor(p["shape"]).to(deca.device)
    pose = torch.zeros_like(torch.tensor(p["pose"])).to(deca.device)          # frontal, jaw closed
    exp = (torch.tensor(p["exp"]) if keep_expression
           else torch.zeros_like(torch.tensor(p["exp"]))).to(deca.device)
    verts = _decode_verts(deca, shape, exp, pose)
    _export_glb(verts, faces, f"{out_dir}/{name}/{name}_neutral.glb", color=color)
    print("exported", f"{name}_neutral.glb")


# --- identity-consistency measurement -----------------------------------------------------
# For a *hash*, the same person must map to the same identity vector across photos. DECA's
# `shape` is a reconstruction objective, not an identity-invariance one, so it drifts; ArcFace
# (a face-recognition embedding) is trained for exactly that invariance, so it's the ceiling.
# Needs: pip install insightface onnxruntime   (opencv/numpy already come from bootstrap)

def _arcface_app(_cache={}):
    """Lazily build + cache an InsightFace buffalo_l app. CPU is fine for embeddings."""
    if "app" not in _cache:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("ArcFace needs InsightFace: pip install insightface onnxruntime") from e
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))   # downloads buffalo_l on first call
        _cache["app"] = app
    return _cache["app"]


def arcface_embed(image_path):
    """512-d, L2-normalized ArcFace identity embedding for the largest face (or None)."""
    import cv2
    img = cv2.imread(image_path) if image_path else None
    if img is None:
        return None
    faces = _arcface_app().get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score).normed_embedding   # already L2-normalized


def _find_input(stem, in_dir):
    import glob
    hits = sorted(glob.glob(os.path.join(in_dir, stem + ".*")))
    return hits[0] if hits else None


def _pairwise(vecs_by_person, metric):
    """All pairwise distances split into same-person (intra) and different-person (inter)."""
    import numpy as np
    import itertools
    items = [(p, np.asarray(v, float).ravel())
             for p, vs in vecs_by_person.items() for v in vs if v is not None]
    persons = [p for p, _ in items]
    X = [v for _, v in items]
    if metric == "cos":
        dist = lambda a, b: 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    else:
        dist = lambda a, b: float(np.linalg.norm(a - b))
    intra, inter = [], []
    for i, j in itertools.combinations(range(len(X)), 2):
        (intra if persons[i] == persons[j] else inter).append(dist(X[i], X[j]))
    return np.array(intra), np.array(inter)


def _auc(intra, inter):
    """P(a same-person pair is closer than a different-person pair). 1.0 = perfect separation."""
    if not len(intra) or not len(inter):
        return float("nan")
    wins = sum(float((a < inter).sum() + 0.5 * (a == inter).sum()) for a in intra)
    return wins / (len(intra) * len(inter))


def consistency_report(person_of, in_dir=IN_DIR, out_dir=OUT_DIR, extractors=("deca", "arcface")):
    """How stable is each extractor's identity vector across photos of the same person?

    person_of: {imagename_stem: person_label}. Need >=2 photos for at least one person so the
    same-person (intra) bucket is non-empty. Prints, per extractor: mean intra/inter identity
    distance, the inter/intra separation ratio (higher = better), and a verification AUC
    (1.0 = same-person pairs always closer than different-person pairs).

    DECA shape (from <stem>_params.npz, L2) is the current carrier; ArcFace buffalo_l (cosine,
    from the original photo in in_dir) is the consistency ceiling to compare against.
    """
    import numpy as np
    by = {}
    for stem, p in person_of.items():
        by.setdefault(p, []).append(stem)

    rows = []
    if "deca" in extractors:
        def deca_vec(s):
            f = f"{out_dir}/{s}/{s}_params.npz"
            return np.load(f)["shape"].ravel() if os.path.exists(f) else None
        intra, inter = _pairwise({p: [deca_vec(s) for s in ss] for p, ss in by.items()}, "l2")
        rows.append(("DECA shape", "l2", intra, inter))
    if "arcface" in extractors:
        def arc_vec(s):
            return arcface_embed(_find_input(s, in_dir))
        intra, inter = _pairwise({p: [arc_vec(s) for s in ss] for p, ss in by.items()}, "cos")
        rows.append(("ArcFace buffalo_l", "cos", intra, inter))

    print(f"{'extractor':<18} {'metric':<6} {'intra':>7} {'inter':>7} {'sep(x)':>7} {'AUC':>6}")
    out = {}
    for name, metric, intra, inter in rows:
        ratio = inter.mean() / (intra.mean() + 1e-9) if len(intra) and len(inter) else float("nan")
        auc = _auc(intra, inter)
        print(f"{name:<18} {metric:<6} {intra.mean():>7.3f} {inter.mean():>7.3f} "
              f"{ratio:>7.2f} {auc:>6.3f}")
        out[name] = {"intra": intra, "inter": inter, "ratio": ratio, "auc": auc}
    return out
