"""Emit a ComfyUI **API-format** prompt JSON (literal values) from a litegraph
builder — the exact payload a RunPod serverless endpoint wants as input.workflow.

ComfyUI's own "Save (API Format)" produces this format; we generate it straight
from build_m2.py so there's no GUI round-trip. Unlike to_swarm.py (which inserts
SwarmUI ${param} placeholders) and to_comfy.py (which emits the litegraph UI
graph), every widget here is a literal value and the shape is the flat
{node_id: {class_type, inputs}} ComfyUI executes.

Usage:  python comfy/to_comfy_api.py   ->  comfy/api/*.json
"""

import json
import pathlib

import build_m2
import build_m2_portrait
from to_swarm import WIDGET_NAMES, build_links_lookup

OUT = pathlib.Path(__file__).with_name("api")


def to_api_prompt(g):
    """litegraph graph -> ComfyUI API prompt: connections become [src_id, slot],
    widgets become literal values (skipping litegraph-only widgets via WIDGET_NAMES)."""
    links = build_links_lookup(g)
    prompt = {}
    for n in g["nodes"]:
        inputs = {}
        for inp in n["inputs"]:                       # connection inputs
            if inp["link"] in links:
                src, slot = links[inp["link"]]
                inputs[inp["name"]] = [str(src), slot]
        for wname, wval in zip(WIDGET_NAMES.get(n["type"], []), n.get("widgets_values", [])):
            if wname is not None:                     # widget inputs (literal)
                inputs[wname] = wval
        prompt[str(n["id"])] = {"inputs": inputs, "class_type": n["type"], "_meta": {"title": n["type"]}}
    return prompt


def convert(builder_module, out_name):
    g = builder_module.build()
    builder_module.validate(g)
    OUT.mkdir(exist_ok=True)
    (OUT / out_name).write_text(json.dumps(to_api_prompt(g), indent=2))
    print(f"wrote api/{out_name}  ({len(g['nodes'])} nodes)")


if __name__ == "__main__":
    convert(build_m2, "M2_instantid_hashed.api.json")
    convert(build_m2_portrait, "M2_portrait_fp16.api.json")
