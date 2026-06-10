"""ComfyUI_FaceHash — inject the Stage-2 ArcFace identity hash into InstantID.

`FaceHashApplyInstantID` is `ApplyInstantID` with one change: between InstantID's
ArcFace extraction and its image-projection, the 512-d identity embedding is run
through `arcface_keymix_v1(embed, key)` (a keyed signed permutation). The rendered
face becomes a deterministic, *different* identity — same person+key -> same face,
every photo. Keypoints (pose/expression) are extracted separately and left
untouched, so feeding the original photo as `image_kps` keeps pose/expression.

Requires the sibling custom node `ComfyUI_InstantID` to be installed.
"""

import glob
import importlib
import os
import sys

import numpy as np
import torch

# Make the sibling custom node importable (.../custom_nodes on sys.path).
_CUSTOM_NODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CUSTOM_NODES not in sys.path:
    sys.path.insert(0, _CUSTOM_NODES)


def _import_instantid():
    """InstantID's folder name varies by installer — cubiq's git clone is
    `ComfyUI_InstantID`, a ComfyUI-Manager install is `comfyui_instantid`. Try the known
    names, then any sibling dir that actually contains InstantID.py, and import that
    (already-loaded) module so our monkeypatch targets the same module object."""
    candidates = ["ComfyUI_InstantID", "comfyui_instantid"]
    for p in glob.glob(os.path.join(_CUSTOM_NODES, "*", "InstantID.py")):
        candidates.append(os.path.basename(os.path.dirname(p)))
    for name in dict.fromkeys(candidates):  # dedupe, preserve order
        try:
            return importlib.import_module(name + ".InstantID")
        except Exception:
            continue
    return None


IID = _import_instantid()
if IID is None:  # pragma: no cover - clear failure if InstantID missing
    raise ImportError(
        "ComfyUI_FaceHash requires the InstantID custom node (cubiq/ComfyUI_InstantID) in "
        "custom_nodes/ — folder name ComfyUI_InstantID or comfyui_instantid."
    )
ApplyInstantID = IID.ApplyInstantID

from .facehash import arcface_keymix_v1


class FaceHashApplyInstantID(ApplyInstantID):
    """Apply InstantID with the reference identity hashed (Stage-2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "key": ("STRING", {"default": "demo-key"}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ip_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "combine_embeds": (["average", "norm average", "concat"], {"default": "average"}),
            },
            "optional": {
                "image_kps": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("MODEL", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "FaceHash"

    def apply(self, key, offset, **kwargs):
        # Temporarily wrap InstantID's extractor: hash the 512-d identity embedding
        # (extract_kps=False) but leave keypoints (extract_kps=True) untouched.
        orig = IID.extractFeatures

        def patched(insightface, image, extract_kps=False):
            out = orig(insightface, image, extract_kps=extract_kps)
            if extract_kps or out is None or not key:
                return out
            arr = arcface_keymix_v1(out.detach().cpu().numpy(), key, offset)  # (N, 512)
            return torch.from_numpy(np.ascontiguousarray(arr)).to(out.device, dtype=out.dtype)

        IID.extractFeatures = patched
        try:
            return super().apply_instantid(**kwargs)
        finally:
            IID.extractFeatures = orig


NODE_CLASS_MAPPINGS = {"FaceHashApplyInstantID": FaceHashApplyInstantID}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceHashApplyInstantID": "Apply InstantID (FaceHash)"}
