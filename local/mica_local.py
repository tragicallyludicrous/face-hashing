"""
mica_local.py — clean, in-process MICA Stage 1 for macOS (CPU/MPS). No subprocess, no
disk round-trip for the ArcFace blob (which is what `demo.py` does via its `-a` temp dir).

    import mica_local as mica
    m   = mica.load(device="cpu")          # build detector + model ONCE
    vec = mica.embed(m, "photo.jpg")       # -> (300,) MICA identity, or None if no face
    mica.reconstruct(m, "in", "out")       # batch -> out/<stem>/{identity.npy, <stem>.glb}

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


class Handle:
    """Loaded detector + model + FLAME faces, ready for embed()/reconstruct()."""
    def __init__(self, mica, app, faces, device):
        self.mica, self.app, self.faces, self.device = mica, app, faces, device


def load(device="cpu"):
    """Build the RetinaFace detector and the MICA model once. Returns a Handle.

    device: "cpu" (default, verified faithful), "mps", or "auto" (mps-if-available).
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
    from utils.landmark_detector import LandmarksDetector, detectors
    from micalib.models.mica import MICA

    cfg = get_cfg_defaults()                       # cfg.mica_dir auto-resolves to MICA_DIR
    app = LandmarksDetector(model=detectors.RETINAFACE)
    mica = MICA(cfg, device)                       # __init__ -> load_model() loads data/pretrained/mica.tar
    mica.eval()
    mica.testing = True                            # skip decode()'s training-only `codedict['flame']` GT block
    return Handle(mica, app, _get_faces(mica), device)


def embed(h, image_path, with_mesh=False):
    """Photo -> 300-d MICA identity (numpy float32). None if no face detected.

    with_mesh=True also returns the neutral FLAME vertices: returns (code, verts).
    Identity depends only on the ArcFace crop; the 224px tensor MICA stores is unused for the code.
    """
    import cv2
    from insightface.app.common import Face
    from datasets.creation.util import get_arcface_input, get_center

    img = cv2.imread(image_path)
    if img is None:
        return None
    bboxes, kpss = h.app.detect(img)
    if bboxes is None or len(bboxes) == 0:
        return None
    i = get_center(bboxes, img)                    # the most central face
    kps = kpss[i] if kpss is not None else None
    face = Face(bbox=bboxes[i, 0:4], kps=kps, det_score=bboxes[i, 4])
    blob, aimg = get_arcface_input(face, img)      # blob: (3,112,112) float32, ArcFace-normalized

    dev = h.device
    arcface = torch.tensor(blob).float().to(dev)[None]                         # (1,3,112,112)
    images = cv2.resize(aimg, (224, 224)).transpose(2, 0, 1) / 255.0
    images = torch.tensor(images).float().to(dev)[None]                        # (1,3,224,224); stored, unused for ID

    with torch.no_grad():
        codedict = h.mica.encode(images, arcface)
        opdict = h.mica.decode(codedict)
    code = opdict["pred_shape_code"][0].detach().cpu().numpy().astype(np.float32)   # (300,)
    if not with_mesh:
        return code
    verts = opdict["pred_canonical_shape_vertices"][0].detach().cpu().numpy()
    return code, verts


def _export_glb(verts, faces, path, color=(180, 200, 180, 255)):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts * 1000.0, faces=faces, process=False)   # m -> mm (matches MICA .ply)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=list(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def reconstruct(h, in_dir, out_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """Every photo in in_dir -> out_dir/<stem>/{identity.npy, <stem>.glb}. Returns the stems done."""
    done = []
    for f in sorted(os.listdir(in_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        out = embed(h, os.path.join(in_dir, f), with_mesh=h.faces is not None)
        if out is None:
            print(f"  {stem}: no face detected — skipped")
            continue
        d = os.path.join(out_dir, stem)
        os.makedirs(d, exist_ok=True)
        if h.faces is not None:
            code, verts = out
            _export_glb(verts, h.faces, os.path.join(d, f"{stem}.glb"))
        else:
            code = out
        np.save(os.path.join(d, "identity.npy"), code)
        print(f"  {stem}: identity {code.shape}" + ("" if h.faces is None else f" + {stem}.glb"))
        done.append(stem)
    return done


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MICA Stage 1 (local, in-process).")
    p.add_argument("-i", "--in_dir", required=True, help="folder of photos")
    p.add_argument("-o", "--out_dir", required=True, help="output folder")
    p.add_argument("--device", default=os.environ.get("MICA_DEVICE", "cpu"),
                   help="cpu (default) | mps | auto")
    a = p.parse_args()
    h = load(device=a.device)
    print(f"MICA loaded on {h.device}; reconstructing {a.in_dir} -> {a.out_dir}")
    names = reconstruct(h, a.in_dir, a.out_dir)
    print(f"done: {len(names)} face(s)")
