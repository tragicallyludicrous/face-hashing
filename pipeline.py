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

⚠️ Two DECA-API details vary by revision and are marked below; fallbacks are inline.
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
        faces = deca.flame.faces_tensor.detach().cpu().numpy()  # ⚠️ verify attr name in Colab
    except AttributeError:
        import trimesh
        faces = trimesh.load(deca_cfg.model.topology_path, process=False).faces  # fallback: template obj
    return deca, faces


def _decode_verts(deca, shape, exp, pose):
    """FLAME params -> mesh vertices, calling the decoder directly (no renderer)."""
    import torch
    with torch.no_grad():
        verts, _, _ = deca.flame(  # ⚠️ verify kwarg names against decalib/models/FLAME.py
            shape_params=shape, expression_params=exp, pose_params=pose,
        )
    return verts


def _export_glb(verts, faces, path, color=(180, 180, 200, 255)):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(), faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=list(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def reconstruct(deca, faces, in_dir=IN_DIR, out_dir=OUT_DIR, detector="fan"):
    """Every photo in in_dir -> out_dir/<name>/{<name>.glb, <name>_params.npz}. Returns the names."""
    import torch
    import numpy as np
    from decalib.datasets import datasets

    data = datasets.TestData(in_dir, iscrop=True, face_detector=detector)  # FAN detector crops to 224
    names = []
    for i in range(len(data)):
        d = data[i]
        name = d["imagename"]
        images = d["image"].to(deca.device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(images)
        verts = _decode_verts(deca, codedict["shape"], codedict["exp"], codedict["pose"])
        _export_glb(verts, faces, f"{out_dir}/{name}/{name}.glb")
        params = {k: v.detach().cpu().numpy()
                  for k, v in codedict.items() if torch.is_tensor(v) and k != "images"}
        np.savez(f"{out_dir}/{name}/{name}_params.npz", **params)
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
