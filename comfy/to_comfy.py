"""Emit STANDALONE ComfyUI workflow JSONs (bare litegraph graph) you can open
directly in raw ComfyUI — e.g. on RunPod, which has no SwarmUI envelope wrapper.

SwarmUI's CustomWorkflows are an envelope ({workflow, prompt, custom_params,...});
raw ComfyUI's "Open" wants the graph itself, with the top-level `id`/`revision` and
per-node `properties` the post-1.0 frontend validates. We reuse to_swarm's enrich
step (which harvests those from a real SwarmUI export) and write just the graph.

Usage:  python comfy/to_comfy.py   ->  comfy/comfyui/*.json
"""

import json
import pathlib

import build_m2
import build_m2_portrait
from to_swarm import enrich_workflow, load_templates

OUT = pathlib.Path(__file__).with_name("comfyui")


def convert(builder_module, out_name):
    _, _, props = load_templates()
    g = builder_module.build()
    builder_module.validate(g)
    w = enrich_workflow(g, props)  # adds id/revision + node properties (cnr_id/ver)
    OUT.mkdir(exist_ok=True)
    out = OUT / out_name
    out.write_text(json.dumps(w, indent=2))
    print(f"wrote comfyui/{out_name}  ({len(w['nodes'])} nodes, id={w['id'][:8]}…)")


if __name__ == "__main__":
    convert(build_m2, "M2_instantid_hashed.json")
    convert(build_m2_portrait, "M2_portrait_fp16.json")
