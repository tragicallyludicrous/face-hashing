"""
mica_local.py — clean, in-process MICA Stage 1 for macOS (CPU/MPS). No subprocess, no
disk round-trip for the ArcFace blob (which is what `demo.py` does via its `-a` temp dir).

    import mica_local as mica
    m   = mica.load(device="cpu")          # build detector + model ONCE
    vec = mica.embed(m, "photo.jpg")       # -> (300,) MICA identity, or None if no face
    arc = mica.arcface_embed(m, "photo.jpg")  # -> (512,) ArcFace embedding (the recognition 'ceiling')
    mica.reconstruct(m, "in", "out")       # batch -> out/<stem>/{identity.npy, arcface.npy, <stem>.glb}

This reuses MICA's OWN code (detector, ArcFace preprocessing, network, FLAME decoder) — it just
drives it in-process with a clean API instead of the `demo.py` two-pass disk flow. It assumes you
have already run `patch_mica_for_mac.py` against the MICA checkout (that applies the on-disk source
fixes: chumpy shim, LandmarksType rename, numpy-2.0 aliases, CUDA->CPU detector provider). The
RUNTIME shims (CUDA-less autocast, CPU-mapped torch.load, MPS env, device) are applied here so this
module doesn't depend on demo.py's injected preamble.

Determinism check: outputs match `demo.py`'s `identity.npy` (cosine ~1.0); demo.py CPU already
matched Colab at cosine 1.0, so this is Stage 1, faithful, as an importable function.
"""
import contextlib
import os
import sys

import numpy as np

# --- locate the patched MICA checkout and put it on sys.path -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MICA_DIR = os.environ.get("MICA_DIR", os.path.join(_HERE, "MICA"))
if MICA_DIR not in sys.path:
    sys.path.insert(0, MICA_DIR)

# --- runtime shims (must precede any MICA import / model construction) ----------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

torch.cuda.amp.autocast = lambda *a, **k: contextlib.nullcontext()   # no CUDA autocast on a Mac
_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu")     # checkpoint was saved from CUDA
    k.setdefault("weights_only", False)     # torch>=2.6 flipped the default; mica.tar has non-tensor payload
    return _orig_load(*a, **k)
torch.load = _cpu_load


def _pick_device(device):
    if device and device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _get_faces(mica):
    """FLAME triangle topology; the attribute path varies slightly by MICA revision."""
    for path in ("flameModel.generator.faces_tensor", "flameModel.faces_tensor", "flame.faces_tensor"):
        obj = mica
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj.detach().cpu().numpy()
        except AttributeError:
            continue
    return None


def _get_flame(mica):
    """The FLAME decoder module inside the loaded MICA model (attribute path varies by revision)."""
    for path in ("flameModel.generator", "flameModel.flame", "flame"):
        obj = mica
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "faces_tensor"):
                return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate the FLAME decoder on the MICA model")


def _faces_list(data):
    """Pull the list of face objects out of loupe's `landmarks --json`, schema-tolerantly."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "left_eye" in data:                             # a single face object at top level
            return [data]
        for k in ("landmarks", "faces", "results", "data"):
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():                            # fallback: any list of face-like dicts
            if isinstance(v, list) and v and isinstance(v[0], dict) and "left_eye" in v[0]:
                return v
    return []


def _detect_vision(image_path, shape):
    """macOS Vision (via the `loupe` CLI) -> (bbox[x1,y1,x2,y2], kps[5,2]) in pixels, or None.

    loupe emits NORMALIZED, top-left-origin coords (it already flips Vision's bottom-left Y), regions
    left_eye/right_eye/nose/outer_lips/... as lists of {x,y}. We build ArcFace's 5-point kps and
    disambiguate left/right by IMAGE x, so Vision's anatomical naming can't mirror the crop.
    """
    import json
    import subprocess
    import numpy as np

    r = subprocess.run(["loupe", "landmarks", "--json", image_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"loupe failed: {r.stderr.strip() or r.stdout.strip()}")
    faces = _faces_list(json.loads(r.stdout))
    if not faces:
        return None
    if not all(k in faces[0] for k in ("box", "left_eye", "right_eye", "nose", "outer_lips")):
        raise RuntimeError("unexpected loupe landmarks JSON; got keys "
                           f"{list(faces[0].keys())} — paste `loupe landmarks --json <photo>` to fix the parser")

    H, W = shape[0], shape[1]
    pts = lambda region: np.array([[p["x"], p["y"]] for p in region], dtype=np.float64)
    bc = lambda f: np.array([f["box"]["x"] + f["box"]["width"]/2, f["box"]["y"] + f["box"]["height"]/2])
    f = min(faces, key=lambda f: float(np.hypot(*(bc(f) - 0.5))))   # most central face

    eyes = sorted([pts(f["left_eye"]).mean(0), pts(f["right_eye"]).mean(0)], key=lambda p: p[0])
    lips = pts(f["outer_lips"])
    kps = np.array([eyes[0], eyes[1], pts(f["nose"]).mean(0),
                    lips[lips[:, 0].argmin()], lips[lips[:, 0].argmax()]], dtype=np.float32)
    kps[:, 0] *= W; kps[:, 1] *= H                                  # normalized -> pixels

    b = f["box"]
    bbox = np.array([b["x"]*W, b["y"]*H, (b["x"]+b["width"])*W, (b["y"]+b["height"])*H], dtype=np.float32)
    return bbox, kps


class Handle:
    """Loaded detector + model + FLAME faces, ready for embed()/reconstruct()."""
    def __init__(self, mica, app, faces, device, detector="antelopev2"):
        self.mica, self.app, self.faces = mica, app, faces
        self.device, self.detector = device, detector


def load(device="cpu", detector="antelopev2"):
    """Build the face detector + MICA model once. Returns a Handle.

    device:   "cpu" (default, verified faithful), "mps", or "auto" (mps-if-available).
    detector: "antelopev2" (InsightFace, default) | "vision" (macOS Vision via the `loupe` CLI —
              drops the 360 MB antelopev2 pack; needs `brew install georgemandis/tap/loupe`).
    """
    device = _pick_device(device)
    if not os.path.isdir(MICA_DIR):
        raise FileNotFoundError(
            f"MICA checkout not found at {MICA_DIR}. Clone it and run patch_mica_for_mac.py "
            f"(see local/README.md), or set MICA_DIR.")
    ckpt = os.path.join(MICA_DIR, "data", "pretrained", "mica.tar")
    flame = os.path.join(MICA_DIR, "data", "FLAME2020", "generic_model.pkl")
    for f in (ckpt, flame):
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing weight file: {f} (see local/README.md §3)")

    from configs.config import get_cfg_defaults
    from micalib.models.mica import MICA

    app = None
    if detector == "antelopev2":
        from utils.landmark_detector import LandmarksDetector, detectors
        app = LandmarksDetector(model=detectors.RETINAFACE)
    elif detector == "vision":
        import shutil
        if shutil.which("loupe") is None:
            raise RuntimeError("--detector vision needs the loupe CLI: brew install georgemandis/tap/loupe")
    elif detector == "none":
        pass                                       # FLAME-decode only (e.g. compose_mesh); no detection
    else:
        raise ValueError(f"unknown detector {detector!r} (use 'antelopev2', 'vision', or 'none')")

    cfg = get_cfg_defaults()                       # cfg.mica_dir auto-resolves to MICA_DIR
    mica = MICA(cfg, device)                       # __init__ -> load_model() loads data/pretrained/mica.tar
    mica.eval()
    mica.testing = True                            # skip decode()'s training-only `codedict['flame']` GT block
    return Handle(mica, app, _get_faces(mica), device, detector)


def _run(h, image_path):
    """Detect -> crop -> encode -> decode once.

    Returns {'code': (300,) MICA identity, 'verts': (5023,3) neutral mesh,
             'arcface': (512,) L2-normalized ArcFace embedding}, or None if no face.
    The 512-d ArcFace vector is the identity signal MICA consumes BEFORE the FLAME-shape
    bottleneck — the recognition 'ceiling'. Identity (code) depends only on it; the 224px
    `images` tensor MICA stores is unused for the embedding.
    """
    import cv2
    from insightface.app.common import Face
    from datasets.creation.util import get_arcface_input

    img = cv2.imread(image_path)
    if img is None:
        return None

    if h.detector == "vision":
        det = _detect_vision(image_path, img.shape)        # macOS Vision via the loupe CLI
        if det is None:
            return None
        bbox, kps = det
        face = Face(bbox=bbox, kps=kps, det_score=1.0)
    else:                                                  # antelopev2 (InsightFace)
        from datasets.creation.util import get_center
        bboxes, kpss = h.app.detect(img)
        if bboxes is None or len(bboxes) == 0:
            return None
        i = get_center(bboxes, img)                        # the most central face
        kps = kpss[i] if kpss is not None else None
        face = Face(bbox=bboxes[i, 0:4], kps=kps, det_score=bboxes[i, 4])

    blob, aimg = get_arcface_input(face, img)              # blob: (3,112,112) float32, ArcFace-normalized

    dev = h.device
    arcface = torch.tensor(blob).float().to(dev)[None]                         # (1,3,112,112)
    images = cv2.resize(aimg, (224, 224)).transpose(2, 0, 1) / 255.0
    images = torch.tensor(images).float().to(dev)[None]                        # (1,3,224,224); unused for ID

    with torch.no_grad():
        codedict = h.mica.encode(images, arcface)
        opdict = h.mica.decode(codedict)
    return {
        "code": opdict["pred_shape_code"][0].detach().cpu().numpy().astype(np.float32),       # (300,)
        "verts": opdict["pred_canonical_shape_vertices"][0].detach().cpu().numpy(),           # (5023,3)
        "arcface": codedict["arcface"][0].detach().cpu().numpy().astype(np.float32),          # (512,) L2-norm
    }


def embed(h, image_path, with_mesh=False):
    """Photo -> 300-d MICA identity (numpy float32). None if no face. with_mesh=True -> (code, verts)."""
    r = _run(h, image_path)
    if r is None:
        return None
    return (r["code"], r["verts"]) if with_mesh else r["code"]


def arcface_embed(h, image_path):
    """Photo -> 512-d ArcFace recognition embedding (MICA's backbone, L2-normalized). None if no face."""
    r = _run(h, image_path)
    return None if r is None else r["arcface"]


def _export_glb(verts, faces, path, color=(180, 200, 180, 255)):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts * 1000.0, faces=faces, process=False)   # m -> mm (matches MICA .ply)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=list(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def reconstruct(h, in_dir, out_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """Every photo in in_dir -> out_dir/<stem>/{identity.npy, arcface.npy, <stem>.glb}. Returns the stems."""
    done = []
    for f in sorted(os.listdir(in_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        r = _run(h, os.path.join(in_dir, f))
        if r is None:
            print(f"  {stem}: no face detected — skipped")
            continue
        d = os.path.join(out_dir, stem)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "identity.npy"), r["code"])
        np.save(os.path.join(d, "arcface.npy"), r["arcface"])
        if h.faces is not None:
            _export_glb(r["verts"], h.faces, os.path.join(d, f"{stem}.glb"))
        print(f"  {stem}: identity{r['code'].shape} + arcface{r['arcface'].shape}"
              + ("" if h.faces is None else f" + {stem}.glb"))
        done.append(stem)
    return done


def compose_mesh(h, identity, deca_npz, out_glb, color=(200, 180, 180, 255)):
    """flame(shape=MICA identity, exp+pose=DECA) -> .glb — the MICA identity wearing the photo's expression/pose.

    identity: a MICA identity.npy path (300-d FLAME shape; raw or hashed), or a numpy array.
    deca_npz: a DECA <stem>_params.npz path (uses its 'exp' and 'pose' groups).

    Identity comes from MICA (consistent), expression+pose from the ORIGINAL photo (DECA) — both decode
    through the same FLAME 2020 basis. SMIRK/EMOCA can later supply better exp/pose: same call, same basis.
    """
    import numpy as np
    flame = _get_flame(h.mica)
    n_exp = int(flame.shapedirs.shape[2]) - 300            # shapedirs = [300 shape | n_exp expression]
    fit = lambda v, n: np.pad(np.asarray(v, np.float32).ravel(), (0, n))[:n]   # zero-pad / truncate to n

    shape = identity if isinstance(identity, np.ndarray) else np.load(identity)
    d = np.load(deca_npz)
    S = torch.tensor(fit(shape, 300))[None].float().to(h.device)
    E = torch.tensor(fit(d["exp"], n_exp))[None].float().to(h.device)
    P = torch.tensor(fit(d["pose"], 6))[None].float().to(h.device)
    with torch.no_grad():
        verts = flame(shape_params=S, expression_params=E, pose_params=P)[0]   # (1, 5023, 3), meters
    _export_glb(verts[0].detach().cpu().numpy(), h.faces, out_glb, color=color)
    print("composed", out_glb, "(MICA shape + DECA exp/pose)")
    return out_glb


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MICA Stage 1 (local, in-process).")
    p.add_argument("-i", "--in_dir", required=True, help="folder of photos")
    p.add_argument("-o", "--out_dir", required=True, help="output folder")
    p.add_argument("--device", default=os.environ.get("MICA_DEVICE", "cpu"),
                   help="cpu (default) | mps | auto")
    p.add_argument("--detector", default="antelopev2", choices=["antelopev2", "vision"],
                   help="antelopev2 (InsightFace, default) | vision (macOS, via loupe)")
    a = p.parse_args()
    h = load(device=a.device, detector=a.detector)
    print(f"MICA loaded on {h.device} (detector: {h.detector}); reconstructing {a.in_dir} -> {a.out_dir}")
    names = reconstruct(h, a.in_dir, a.out_dir)
    print(f"done: {len(names)} face(s)")
