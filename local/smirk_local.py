"""
smirk_local.py — clean, in-process SMIRK on macOS (CPU/MPS), renderer-free. The Stage-3 sibling of
mica_local.py: MICA gives the consistent *identity* (shape); SMIRK gives the *expression/pose/jaw/
eyelids* from the original photo. Compose them and you get the (possibly hashed) identity wearing the
photo's expression — the input to Stage 4 (photoreal + composite).

SMIRK (CVPR'24, MIT) reconstructs much better expressions than DECA. Its INFERENCE path — the timm
encoder + its FLAME (lbs) — is pytorch3d-free; only SMIRK's Renderer needs pytorch3d. So, exactly
like mica_local vs MICA's demo.py, we import SmirkEncoder + FLAME directly and never touch the
Renderer.

    import smirk_local as smirk
    h   = smirk.load(device="cpu")              # build encoder + FLAME ONCE
    p   = smirk.params(h, "photo.jpg")          # -> {shape,exp,pose,jaw,eyelid,cam} numpy dict, or None
    smirk.compose(h, "photo.jpg", "identity.npy", "out.glb")   # MICA shape + SMIRK exp/pose -> .glb
    smirk.reconstruct(h, "in", "out", mica_dir="out/MICA")     # batch -> per-stem npz + glb (+composed)

CLI mirrors mica_local — batch by default, single-image compose when you pass one photo:

    python smirk_local.py -i in -o out/SMIRK --mica out/MICA      # batch: native + composed glb
    python smirk_local.py photo.jpg --mica identity.npy -o out.glb  # one photo, swap identity in

SETUP (reuse local/.venv — it has torch + a chumpy shim FLAME's pkl needs):
    pip install timm mediapipe
    git clone https://github.com/georgeretsi/smirk.git local/smirk
    cd local/smirk && bash quick_install.sh     # SMIRK_em1.pt + FLAME/eyelid/mediapipe assets
    # (if quick_install's FLAME unzips double-nested, flatten so generic_model.pkl sits directly in
    #  local/smirk/assets/FLAME2020/)

DETECTOR: SMIRK ships a mediapipe FaceLandmarker, which misses small-in-frame and profile faces that
MICA's RetinaFace catches. We try mediapipe first (no change for the easy majority) and, on a miss,
fall back to the same antelopev2/RetinaFace detector MICA uses (already in ~/.insightface/models) to
locate the face, then re-run mediapipe on a padded crop. So smirk_local detects the same faces
mica_local does, with mediapipe's native crop framing preserved.

NOTE: load() does os.chdir(SMIRK_DIR) because SMIRK's FLAME and mediapipe_utils load hardcoded
relative 'assets/...' paths. All public functions resolve image/identity/in/out to absolute paths
first, so that chdir is transparent to callers.
"""
import os
import sys
import warnings

import numpy as np

# SMIRK targets old numpy; restore the aliases numpy 2.0 removed, BEFORE importing SMIRK source
# (src/FLAME/FLAME.py does `np.float = np.float_`, and np.float_ itself is gone in 2.x). Silence the
# legacy-name FutureWarning the hasattr probe trips on a couple of these names.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str),
                   ("complex", complex), ("float_", np.float64), ("complex_", np.complex128),
                   ("unicode_", str), ("Inf", np.inf), ("Infinity", np.inf), ("NaN", np.nan),
                   ("NAN", np.nan), ("infty", np.inf)):
        if not hasattr(np, _n):
            setattr(np, _n, _v)

_HERE = os.path.dirname(os.path.abspath(__file__))
SMIRK_DIR = os.environ.get("SMIRK_DIR", os.path.join(_HERE, "smirk"))

# SMIRK's encoder param groups (the Stage-3 payload). pose_params is GLOBAL head rotation only (3);
# jaw is separate (3); eyelids are SMIRK's own additive blendshapes (2). This is the full set the
# encoder emits and that SMIRK's FLAME.forward consumes — we serialize exactly these keys.
PARAM_KEYS = ("shape_params", "expression_params", "pose_params", "jaw_params", "eyelid_params", "cam")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

_orig_load = torch.load
def _cpu_load(*a, **k):                      # SMIRK_em1.pt was saved from CUDA
    k.setdefault("map_location", "cpu")
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _cpu_load


def _pick_device(d):
    if d and d != "auto":
        return d
    return "mps" if torch.backends.mps.is_available() else "cpu"


def crop_face(frame, landmarks, scale=1.4, image_size=224):
    """SMIRK's similarity-transform crop (copied verbatim from demo.py)."""
    from skimage.transform import estimate_transform
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src = np.array([[center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]])
    dst = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src, dst)


class Handle:
    """Loaded SMIRK encoder + its FLAME decoder, ready for params()/compose()/reconstruct()."""
    def __init__(self, encoder, flame, device):
        self.encoder, self.flame, self.device = encoder, flame, device
        self._rf = "unset"                       # lazily-built RetinaFace fallback detector (see _encode)


def load(device="cpu", checkpoint=None):
    """Build SMIRK's encoder + FLAME once (renderer-free). chdir's into SMIRK_DIR for its asset paths.

    device: "cpu" (default) | "mps" | "auto" (mps-if-available).
    checkpoint: SMIRK .pt (default: pretrained_models/SMIRK_em1.pt under SMIRK_DIR).
    """
    device = _pick_device(device)
    if not os.path.isdir(SMIRK_DIR):
        raise FileNotFoundError(
            f"SMIRK checkout not found at {SMIRK_DIR} — git clone https://github.com/georgeretsi/smirk.git "
            f"local/smirk && (cd local/smirk && bash quick_install.sh), or set SMIRK_DIR.")
    if SMIRK_DIR not in sys.path:
        sys.path.insert(0, SMIRK_DIR)
    os.chdir(SMIRK_DIR)                       # SMIRK's FLAME + mediapipe_utils load hardcoded relative assets/

    ckpt_path = checkpoint
    if not ckpt_path:
        for alt in ("pretrained_models/SMIRK_em1.pt", "trained_models/SMIRK_em1.pt"):
            if os.path.exists(os.path.join(SMIRK_DIR, alt)):
                ckpt_path = os.path.join(SMIRK_DIR, alt); break
    needed = {
        "checkpoint": ckpt_path,
        "FLAME model": os.path.join(SMIRK_DIR, "assets", "FLAME2020", "generic_model.pkl"),
        "mediapipe task": os.path.join(SMIRK_DIR, "assets", "face_landmarker.task"),
        "eyelid blendshape": os.path.join(SMIRK_DIR, "assets", "l_eyelid.npy"),
    }
    for label, path in needed.items():
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"missing SMIRK {label}: {path or '(not found)'} — run quick_install.sh "
                                    f"in {SMIRK_DIR} (see the module docstring SETUP).")

    from src.smirk_encoder import SmirkEncoder
    from src.FLAME.FLAME import FLAME

    enc = SmirkEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("smirk_encoder.", ""): v
                         for k, v in ckpt.items() if "smirk_encoder" in k})
    enc.eval()

    flame = FLAME().to(device)
    flame.eval()
    return Handle(enc, flame, device)


def _retinaface(h):
    """Lazily build the antelopev2/RetinaFace detector MICA uses (~/.insightface/models/antelopev2).

    Only constructed the first time mediapipe misses a face; if insightface/antelopev2 isn't available
    we cache None and silently skip the fallback (mediapipe-only behavior).
    """
    if h._rf == "unset":
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="antelopev2", allowed_modules=["detection"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))      # larger det_size than MICA's -> small faces
            h._rf = app
        except Exception as e:                               # pack missing / insightface absent
            print("  (RetinaFace fallback unavailable:", e, ")")
            h._rf = None
    return h._rf


def _retinaface_landmarks(h, img, run_mediapipe):
    """Faces mediapipe misses (small-in-frame / profile): RetinaFace locates the face, then we re-run
    mediapipe on a padded crop and offset its 478 landmarks back to full-image coords. None if no face."""
    app = _retinaface(h)
    if app is None:
        return None
    faces = app.get(img)                                     # cv2 BGR, like MICA
    if not faces:
        return None
    fc = max(faces, key=lambda x: x.det_score)
    x1, y1, x2, y2 = fc.bbox
    H, W = img.shape[:2]
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 2.0   # generous square crop
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    mp = run_mediapipe(img[b:d, a:c])
    if mp is None:
        return None
    mp = mp.copy(); mp[:, 0] += a; mp[:, 1] += b             # crop coords -> full-image coords
    return mp


def _encode(h, image_path):
    """Image -> SMIRK encoder output dict (tensors on h.device), or None if no face detected."""
    import cv2
    from skimage.transform import warp
    from utils.mediapipe_utils import run_mediapipe

    img = cv2.imread(image_path)
    if img is None:
        return None
    kpt = run_mediapipe(img)
    if kpt is None:
        kpt = _retinaface_landmarks(h, img, run_mediapipe)   # small/profile faces mediapipe alone misses
    if kpt is None:
        return None
    tform = crop_face(img, kpt, scale=1.4, image_size=224)
    crop = warp(img, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(h.device)
    with torch.no_grad():
        return h.encoder(t)


def params(h, image_path):
    """Photo -> SMIRK FLAME params as a numpy dict, or None if no face.

    Keys (PARAM_KEYS): shape_params(300), expression_params(50), pose_params(3 global rot),
    jaw_params(3), eyelid_params(2), cam(3). shape is SMIRK's OWN identity (we usually replace it with
    MICA's in compose); the rest is the per-photo expression/pose payload that drives Stage 3.
    """
    out = _encode(h, image_path)
    if out is None:
        return None
    return {k: out[k][0].detach().cpu().numpy().astype(np.float32) for k in PARAM_KEYS if k in out}


def _decode(h, param_tensors):
    """Run SMIRK's FLAME on a dict of param tensors -> (5023,3) verts numpy."""
    with torch.no_grad():
        fl = h.flame.forward(param_tensors)
    return fl["vertices"][0].detach().cpu().numpy()


def _faces(flame):
    f = getattr(flame, "faces_tensor", None)
    if f is None:
        f = getattr(flame, "faces", None)
    if f is None:
        raise AttributeError("SMIRK FLAME has no faces_tensor/faces attribute")
    return f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)


def _export_glb(verts, faces, path, color=(200, 180, 180, 255)):
    import trimesh
    m = trimesh.Trimesh(vertices=verts * 1000.0, faces=faces, process=False)   # m -> mm (matches MICA .glb)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=list(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m.export(path)


def compose(h, image_path, mica_identity, out_glb, color=(200, 180, 180, 255)):
    """MICA identity + SMIRK expression/pose -> .glb, decoded through SMIRK's FLAME (eyelids/jaw kept).

    image_path:    the original photo (SMIRK reads its expression/pose/jaw/eyelids).
    mica_identity: a MICA identity.npy path (300-d FLAME shape; raw or hashed) or a numpy array.

    The shape basis is shared FLAME 2020, so MICA's shape code drops straight into SMIRK's FLAME. We
    decode through SMIRK's OWN FLAME (not MICA's) so its eyelid/jaw params apply natively.
    """
    image_path = os.path.abspath(image_path)
    out = _encode(h, image_path)
    if out is None:
        print("  no face detected:", image_path); return None
    shape = mica_identity if isinstance(mica_identity, np.ndarray) else np.load(os.path.abspath(mica_identity))
    shape = np.pad(np.asarray(shape, np.float32).ravel(), (0, 300))[:300]
    out["shape_params"] = torch.tensor(shape)[None].float().to(h.device)   # swap MICA identity in
    _export_glb(_decode(h, out), _faces(h.flame), os.path.abspath(out_glb), color=color)
    print("  composed", out_glb, "(MICA shape + SMIRK exp/pose)")
    return out_glb


def reconstruct(h, in_dir, out_dir, mica_dir=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """Every photo in in_dir -> out_dir/<stem>/{smirk_params.npz, smirk.glb}. Returns the stems.

    smirk_params.npz holds PARAM_KEYS (the per-photo expression/pose payload). smirk.glb is SMIRK's
    OWN native reconstruction (its own shape). If mica_dir is given and mica_dir/<stem>/identity.npy
    exists, also writes out_dir/<stem>/composed.glb = that MICA identity wearing this photo's pose.
    """
    in_dir, out_dir = os.path.abspath(in_dir), os.path.abspath(out_dir)
    mica_dir = os.path.abspath(mica_dir) if mica_dir else None
    faces, done = _faces(h.flame), []
    for f in sorted(os.listdir(in_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        out = _encode(h, os.path.join(in_dir, f))
        if out is None:
            print(f"  {stem}: no face detected — skipped"); continue
        d = os.path.join(out_dir, stem)
        os.makedirs(d, exist_ok=True)
        npd = {k: out[k][0].detach().cpu().numpy().astype(np.float32) for k in PARAM_KEYS if k in out}
        np.savez(os.path.join(d, "smirk_params.npz"), **npd)
        _export_glb(_decode(h, out), faces, os.path.join(d, "smirk.glb"))   # SMIRK's own shape+exp+pose
        msg = f"  {stem}: smirk_params.npz + smirk.glb"
        if mica_dir:
            idp = os.path.join(mica_dir, stem, "identity.npy")
            if os.path.exists(idp):
                out["shape_params"] = torch.tensor(np.load(idp))[None].float().to(h.device)
                _export_glb(_decode(h, out), faces, os.path.join(d, "composed.glb"))
                msg += " + composed.glb"
            else:
                msg += "  (no MICA identity)"
        print(msg)
        done.append(stem)
    return done


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SMIRK expression/pose (local, in-process). "
                                             "Batch by default; pass one photo for single-image compose.")
    ap.add_argument("image", nargs="?", help="single photo -> compose with --mica into -o .glb")
    ap.add_argument("-i", "--in_dir", help="batch: folder of photos")
    ap.add_argument("-o", "--out", help="single: output .glb  |  batch: output folder")
    ap.add_argument("--mica", help="single: MICA identity.npy  |  batch: MICA out dir (per-stem identity.npy)")
    ap.add_argument("--device", default=os.environ.get("SMIRK_DEVICE", "cpu"), help="cpu (default) | mps | auto")
    ap.add_argument("--checkpoint", default=None, help="SMIRK checkpoint (default: pretrained_models/SMIRK_em1.pt)")
    a = ap.parse_args()

    if a.image:                                  # single-image compose
        if not a.mica or not a.out:
            ap.error("single-image mode needs --mica <identity.npy> and -o <out.glb>")
        img, mica, out = os.path.abspath(a.image), os.path.abspath(a.mica), os.path.abspath(a.out)
        h = load(device=a.device, checkpoint=a.checkpoint)
        print(f"SMIRK loaded on {h.device}; composing one photo")
        compose(h, img, mica, out)
    else:                                        # batch reconstruct
        if not a.in_dir or not a.out:
            ap.error("batch mode needs -i <in_dir> and -o <out_dir> (or pass a single photo)")
        in_dir, out_dir = os.path.abspath(a.in_dir), os.path.abspath(a.out)
        mica = os.path.abspath(a.mica) if a.mica else None
        h = load(device=a.device, checkpoint=a.checkpoint)
        print(f"SMIRK loaded on {h.device}; reconstructing {a.in_dir} -> {a.out}"
              + (f" (+ compose vs {a.mica})" if mica else ""))
        names = reconstruct(h, in_dir, out_dir, mica_dir=mica)
        print(f"done: {len(names)} face(s)")
