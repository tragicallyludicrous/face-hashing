"""Emit STANDALONE ComfyUI workflow JSONs (bare litegraph graph) you can open
directly in raw ComfyUI — e.g. on RunPod, which has no SwarmUI envelope wrapper.

The post-1.0 ComfyUI frontend rejects (blank-canvases) a workflow without a top-level
`id`/`revision`, and wants per-node `properties` with a registry id. We add `id`/`revision`
and a small hardcoded properties map (the exact shape that loads cleanly on RunPod) — so
this is self-contained: no SwarmUI template needed.

Usage:  python comfy/to_comfy.py   ->  comfy/comfyui/*.json
"""

import json
import pathlib
import uuid

import build_m2
import build_m2_portrait

OUT = pathlib.Path(__file__).with_name("comfyui")

# Per-class litegraph node `properties` (registry id + version). Everything not listed
# is comfy-core; FaceHash is a local node with no registry id. These are the exact values
# that load without the "missing/blank" frontend behavior.
_PROPS = {
    "InstantIDModelLoader":  {"cnr_id": "comfyui_instantid", "ver": "72495e806bc2ab9c41581e15ccaa1bcf83c477e8"},
    "InstantIDFaceAnalysis": {"cnr_id": "comfyui_instantid", "ver": "72495e806bc2ab9c41581e15ccaa1bcf83c477e8"},
    "FaceHashApplyInstantID": {},  # local node — no registry id (matches what loads)
}


def _node_props(ntype):
    base = _PROPS.get(ntype, {"cnr_id": "comfy-core", "ver": "0.24.0"})
    return {**base, "Node name for S&R": ntype}


def convert(builder_module, out_name):
    g = builder_module.build()
    builder_module.validate(g)
    g["id"] = str(uuid.uuid4())   # post-1.0 frontend needs these or it renders a blank canvas
    g["revision"] = 0
    for n in g["nodes"]:
        n["properties"] = _node_props(n["type"])
    OUT.mkdir(exist_ok=True)
    out = OUT / out_name
    out.write_text(json.dumps(g, indent=2))
    print(f"wrote comfyui/{out_name}  ({len(g['nodes'])} nodes, id={g['id'][:8]}…)")


if __name__ == "__main__":
    convert(build_m2, "M2_instantid_hashed.json")
    convert(build_m2_portrait, "M2_portrait_fp16.json")
