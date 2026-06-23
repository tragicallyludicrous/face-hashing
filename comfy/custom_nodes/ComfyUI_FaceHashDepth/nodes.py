"""
ComfyUI_FaceHashDepth — Phase B: a single photo IMAGE + key -> aligned, hash-clean depth map
(MICA-derived geometry) for the Stage-4 depth ControlNet, generated ON THE POD.

Architecture: MICA and SMIRK ship colliding top-level packages (utils/configs/datasets) and CANNOT
share one interpreter (see CLAUDE.md / run.py). So this node is an ORCHESTRATOR — it shells out to
the two validated CLIs, exactly like run.py:

    photo + key --[MICA: hash_shape.py]--> hashed 300-d shape.npy
    photo + shape --[SMIRK: render_cond.py --maps]--> <stem>_depth.png + _mask.png (aligned)

then loads the depth/mask back as ComfyUI IMAGE/MASK. Results are cached by (image, key, offset,
shape_source) so repeated runs / key sweeps after the first render are instant.

CONSISTENCY: the geometry hash here and the texture hash in FaceHashApplyInstantID both use the same
`arcface_keymix_v1(key)` but on DIFFERENT backbones (MICA's ArcFace vs antelopev2), so they aren't the
same person's co-designed face — but each is deterministic per key, so the COMBINATION is consistent
per key. **Set this node's `key` to the SAME value as the InstantID FaceHash node.**
"""
import os
import sys
import glob
import json
import time
import hashlib
import tempfile
import subprocess

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    """config.json next to this file; env vars override. See config.example.json."""
    cfg = {}
    p = os.path.join(HERE, "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg = json.load(f)
    cfg.setdefault("python_bin", os.environ.get("FACEHASHDEPTH_PYTHON", sys.executable))
    cfg.setdefault("local_dir", os.environ.get("FACEHASHDEPTH_LOCAL", ""))
    cfg.setdefault("neutral_shape",
                   os.environ.get("FACEHASHDEPTH_NEUTRAL",
                                  os.path.join(cfg.get("local_dir", ""), "neutral_shape.npy")))
    cfg.setdefault("cache_dir", os.environ.get("FACEHASHDEPTH_CACHE", os.path.join(HERE, ".cache")))
    cfg.setdefault("default_device", os.environ.get("FACEHASHDEPTH_DEVICE", "cuda"))
    cfg.setdefault("timeout", int(os.environ.get("FACEHASHDEPTH_TIMEOUT", "900")))
    return cfg


# ---- ComfyUI tensor <-> PNG (node runs in ComfyUI's env: numpy/torch/PIL only) ----
def _image_to_png(image, path):
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _png_to_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _png_to_mask(path):
    arr = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _run(cmd, cwd, timeout):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            "FaceHashDepth subprocess failed:\n  " + " ".join(cmd)
            + "\n--- stdout ---\n" + r.stdout[-2000:]
            + "\n--- stderr ---\n" + r.stderr[-2000:]
        )
    return r.stdout


class FaceHashDepth:
    """Photo IMAGE + key -> aligned depth IMAGE + mask MASK (MICA geometry, SMIRK pose)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shape_source": (["hashed", "own", "neutral"], {"default": "hashed"}),
                "key": ("STRING", {"default": "zack-secret"}),
                "offset": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                "force_rerender": ("BOOLEAN", {"default": False}),
                # MUST match the InstantID FaceHash node (same transform + key). whitened_v3
                # (recommended): on-manifold DERIVED synthetic id from a basis (build_basis.py on
                # synthetic faces); this node reads the basis 'mica' column, InstantID 'antelope'.
                # blend_v2: real-identity gallery (kept for completeness). Same basis_path/gallery_path
                # on both nodes.
                "transform": (["arcface_keymix_v1", "arcface_keymix_whitened_v3", "arcface_blend_v2"],
                              {"default": "arcface_keymix_v1"}),
                "basis_path": ("STRING", {"default": ""}),
                "gallery_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "n_mix": ("INT", {"default": 2, "min": 1, "max": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth", "mask")
    FUNCTION = "generate"
    CATEGORY = "FaceHash"

    def generate(self, image, shape_source, key, offset, device="cuda", force_rerender=False,
                 transform="arcface_keymix_v1", basis_path="", gallery_path="", strength=0.6, n_mix=2):
        cfg = _load_config()
        local_dir = cfg["local_dir"]
        if not local_dir or not os.path.isdir(local_dir):
            raise RuntimeError(
                "FaceHashDepth: set 'local_dir' in "
                f"{os.path.join(HERE, 'config.json')} to the folder holding hash_shape.py + "
                "render_cond.py (the repo's local/). See README.md."
            )
        py = cfg["python_bin"]
        dev = device or cfg["default_device"]
        timeout = cfg["timeout"]

        # ---- cache key: image bytes + the identity-defining params ----
        img_sha = hashlib.sha1((image[0].cpu().numpy() * 255).astype(np.uint8).tobytes()).hexdigest()[:16]
        if shape_source == "hashed":
            if transform == "arcface_keymix_whitened_v3":
                tag = "w_%s_%s_o%+0.2f" % (_safe(key), _safe(os.path.basename(basis_path)), offset)
            elif transform == "arcface_blend_v2":
                tag = "b_%s_%s_s%0.2f_m%d" % (_safe(key), _safe(os.path.basename(gallery_path)), strength, n_mix)
            else:
                tag = "h_%s_%+0.2f" % (_safe(key), offset)
        else:
            tag = shape_source
        ck = "%s__%s__%s" % (img_sha, tag, dev)
        cache = cfg["cache_dir"]
        os.makedirs(cache, exist_ok=True)
        c_depth = os.path.join(cache, ck + "_depth.png")
        c_mask = os.path.join(cache, ck + "_mask.png")
        if not force_rerender and os.path.exists(c_depth) and os.path.exists(c_mask):
            return (_png_to_image(c_depth), _png_to_mask(c_mask))

        with tempfile.TemporaryDirectory() as tmp:
            photo = os.path.join(tmp, "photo.png")          # absolute -> survives SMIRK's chdir
            _image_to_png(image, photo)

            shape_arg = []
            if shape_source == "neutral":
                ns = cfg["neutral_shape"]
                if not os.path.exists(ns):
                    raise RuntimeError(f"FaceHashDepth: neutral_shape not found: {ns}")
                shape_arg = ["--shape", ns]
            elif shape_source == "hashed":
                # MICA interpreter: photo + key -> hashed 300-d shape.npy
                cmd = [py, "hash_shape.py", photo, "--keys", key, "--offset", str(offset),
                       "-o", tmp, "--device", dev, "--transform", transform]
                if transform == "arcface_keymix_whitened_v3":
                    if not basis_path:
                        raise RuntimeError(
                            "FaceHashDepth: arcface_keymix_whitened_v3 needs basis_path (build_basis.py); "
                            "use the SAME basis as the InstantID node."
                        )
                    # MICA geometry path reads the basis 'mica' column (texture uses 'antelope')
                    cmd += ["--basis", basis_path, "--column", "mica"]
                elif transform == "arcface_blend_v2":
                    if not gallery_path:
                        raise RuntimeError(
                            "FaceHashDepth: arcface_blend_v2 needs gallery_path (build_gallery.py); "
                            "use the SAME gallery as the InstantID node."
                        )
                    cmd += ["--gallery", gallery_path, "--column", "mica",
                            "--strength", str(strength), "--n-mix", str(n_mix)]
                _run(cmd, cwd=local_dir, timeout=timeout)
                shp = glob.glob(os.path.join(tmp, "photo__*.npy"))
                if not shp:
                    raise RuntimeError("FaceHashDepth: hash_shape produced no shape (no face?)")
                shape_arg = ["--shape", shp[0]]
            # shape_source == "own": no --shape -> SMIRK's own reconstruction

            # SMIRK interpreter: render aligned depth + mask (absolute out-dir dodges the chdir trap)
            _run([py, "render_cond.py", photo, "--maps", "--out-dir", tmp, "--device", dev] + shape_arg,
                 cwd=local_dir, timeout=timeout)

            out_depth = os.path.join(tmp, "photo_depth.png")
            out_mask = os.path.join(tmp, "photo_mask.png")
            if not os.path.exists(out_depth):
                raise RuntimeError(
                    "FaceHashDepth: render produced no depth — no face detected, or the mesh "
                    "projected outside the frame. Check the input image."
                )
            Image.open(out_depth).save(c_depth)
            if os.path.exists(out_mask):
                Image.open(out_mask).save(c_mask)
            else:                                            # mask optional; emit empty if missing
                Image.new("L", Image.open(out_depth).size, 0).save(c_mask)
            return (_png_to_image(c_depth), _png_to_mask(c_mask))


def _safe(s):
    """Filesystem-safe token for the cache name (the HASH still uses the raw key, never this)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[:40]


NODE_CLASS_MAPPINGS = {"FaceHashDepth": FaceHashDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceHashDepth": "FaceHash Depth (MICA→SMIRK)"}
