"""Convert a litegraph workflow (from build_m2 / build_m2_portrait) into a SwarmUI
CustomWorkflow *envelope* and write it straight into SwarmUI's watched folder, so
edits show up in SwarmUI without a manual import — adjustable params and all.

SwarmUI doesn't store raw litegraph JSON; its CustomWorkflows are an envelope:

  { workflow, prompt, custom_params, param_values, image, description, enable_in_simple }

  - workflow      : the litegraph graph (what the Comfy editor draws) — passed through
  - prompt        : the API-format graph SwarmUI actually runs, with widget values
                    swapped for ${param} placeholders
  - custom_params : the param definitions that become UI fields
  - param_values  : their current values

Reconstructing SwarmUI's ~25-field param entries by hand is brittle, so instead we
harvest them from an existing SwarmUI export (M2-test.json) that already covers every
node type these workflows use — it IS SwarmUI's own output, so it always matches the
running SwarmUI's schema. We just retarget ids/defaults onto the new graph.

Usage:  python comfy/to_swarm.py          # convert both M2 builders -> SwarmUI
"""

import copy
import json
import pathlib
import re
import uuid

import build_m2
import build_m2_portrait

SWARM_CW = pathlib.Path(
    "/Users/hudson/Zack DeZon Dropbox/Team Folder/Home Folders/Documents/"
    "Technical/AI/ComfyUI/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/CustomWorkflows"
)
TEMPLATE = SWARM_CW / "M2-test.json"  # known-good SwarmUI export = param-definition library

# litegraph widgets_values order -> API input names. None = a litegraph-only widget
# with no API input (LoadImage's upload button, KSampler's control_after_generate).
WIDGET_NAMES = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CLIPTextEncode": ["text"],
    "LoadImage": ["image", None],
    "LoadImageMask": ["image", "channel"],
    "InstantIDModelLoader": ["instantid_file"],
    "InstantIDFaceAnalysis": ["provider"],
    "ControlNetLoader": ["control_net_name"],
    "FaceHashApplyInstantID":
        ["key", "offset", "ip_weight", "cn_strength", "start_at", "end_at", "noise", "combine_embeds"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEEncode": [],
    "SetLatentNoiseMask": [],
    "VAEDecode": [],
    "SaveImage": ["filename_prefix"],
}

# (class_type, widget) -> (param_id, placeholder_id, numeric). SwarmUI *core* params.
# Numerics get wrapped in SwarmUI's %%_COMFYFIXME_..._ENDFIXME_%% cast markers.
# cfg is special: placeholder id `cfg_scale`, but the param id is `cfgscale`.
STANDARD = {
    ("CheckpointLoaderSimple", "ckpt_name"): ("model", "model", False),
    ("LoadImage", "image"): ("initimage", "initimage", False),
    ("KSampler", "seed"): ("seed", "seed", True),
    ("KSampler", "steps"): ("steps", "steps", True),
    ("KSampler", "cfg"): ("cfgscale", "cfg_scale", True),
    ("KSampler", "sampler_name"): ("sampler", "sampler", False),
    ("KSampler", "scheduler"): ("scheduler", "scheduler", False),
    ("SaveImage", "filename_prefix"): ("prefix", "prefix", False),
}


def num_ph(phid, default):
    """SwarmUI numeric placeholder, with the COMFYFIXME cast markers."""
    return "%%_COMFYFIXME_${" + phid + ":" + default + "}_ENDFIXME_%%"


def txt_ph(phid, default):
    return "${" + phid + ":" + default + "}"

# Widgets to leave as literal values in the prompt (no param, no placeholder).
LITERAL = {("EmptyLatentImage", "width"), ("EmptyLatentImage", "height"),
           ("EmptyLatentImage", "batch_size")}


def _fmt(v):
    """Default-value formatting matching SwarmUI: 1.0 -> '1', 0.8 -> '0.8'."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _strip_ext(name):
    return re.sub(r"\.(safetensors|ckpt|pt|bin)$", "", name)


def load_templates():
    """Index an existing SwarmUI export by param id (standard) and (class, widget) (auto),
    plus a class -> litegraph node-`properties` map (cnr_id/ver) the frontend wants."""
    if not TEMPLATE.exists():
        raise SystemExit(
            f"Missing param template: {TEMPLATE}\n"
            "Import any FaceHash workflow into SwarmUI once (it writes this file); "
            "to_swarm.py reuses its param definitions."
        )
    doc = json.loads(TEMPLATE.read_text())
    cp = doc["custom_params"]
    std, auto = {}, {}
    for pid, entry in cp.items():
        grp = entry.get("group") or {}
        m = re.match(r"(.+?) \(Node \d+\)$", grp.get("name", ""))
        if m:  # auto-param: group name carries the class, entry name is the widget
            auto[(m.group(1), entry["name"])] = entry
        else:  # standard core param, keyed by its clean id
            std[pid] = entry
    props = {n["type"]: n.get("properties", {}) for n in doc["workflow"]["nodes"]}
    return std, auto, props


def enrich_workflow(g, props):
    """Make the litegraph graph frontend-valid: the post-1.0 ComfyUI frontend wants a
    workflow `id` + `revision`, and node `properties` carrying cnr_id/ver. We harvest the
    properties from the SwarmUI template by class; unknown classes default to comfy-core."""
    g = copy.deepcopy(g)
    g["id"] = str(uuid.uuid4())
    g["revision"] = 0
    for n in g["nodes"]:
        base = props.get(n["type"])
        n["properties"] = dict(base) if base else {
            "cnr_id": "comfy-core", "ver": "0.24.0", "Node name for S&R": n["type"]}
    return g


def build_links_lookup(g):
    return {lid: (src, sslot) for lid, src, sslot, *_ in g["links"]}


def to_envelope(g, std_tpl, auto_tpl, props, description=""):
    links = build_links_lookup(g)
    by_id = {n["id"]: n for n in g["nodes"]}

    # which CLIPTextEncode feeds a `positive` vs `negative` input -> prompt / negativeprompt
    clip_role = {}  # src_node_id -> "prompt" | "negativeprompt"
    for n in g["nodes"]:
        for inp in n["inputs"]:
            if inp["name"] in ("positive", "negative") and inp["link"] in links:
                src = links[inp["link"]][0]
                if by_id[src]["type"] == "CLIPTextEncode":
                    clip_role[src] = "prompt" if inp["name"] == "positive" else "negativeprompt"

    prompt, custom_params, param_values = {}, {}, {}

    def add_param(entry, pid, default, placeholder_type, group=None):
        e = copy.deepcopy(entry)
        e["id"] = pid
        e["default"] = default
        if group:
            e["group"] = group
            e["description"] = f"The {entry['name']} input for {group['name']} ({placeholder_type})"
        custom_params[pid] = e
        param_values[pid] = default

    for n in g["nodes"]:
        ntype = n["type"]
        nid = n["id"]
        inputs = {}

        # connection inputs (resolve link -> [src_id_str, src_slot])
        for inp in n["inputs"]:
            if inp["link"] in links:
                src, sslot = links[inp["link"]]
                inputs[inp["name"]] = [str(src), sslot]

        # widget inputs
        names = WIDGET_NAMES.get(ntype, [])
        for wname, wval in zip(names, n.get("widgets_values", [])):
            if wname is None:
                continue
            key = (ntype, wname)

            if key in LITERAL:
                inputs[wname] = wval
            elif key in STANDARD:
                pid, phid, numeric = STANDARD[key]
                if pid == "model":
                    inputs[wname] = txt_ph(phid, wval)  # keep .safetensors in the placeholder
                    if pid in std_tpl:
                        add_param(std_tpl[pid], pid, _strip_ext(wval), "model")
                else:
                    inputs[wname] = num_ph(phid, _fmt(wval)) if numeric else txt_ph(phid, _fmt(wval))
                    if pid in std_tpl:
                        add_param(std_tpl[pid], pid, wval, std_tpl[pid]["type"])
            elif ntype == "CLIPTextEncode" and wname == "text":
                pid = clip_role.get(nid, "prompt")  # ${prompt} / ${negativeprompt}, no default
                inputs[wname] = "${" + pid + "}"
                if pid in std_tpl:
                    add_param(std_tpl[pid], pid, wval, "text")
            else:
                # auto-param: reuse SwarmUI's own definition for this (class, widget)
                tpl = auto_tpl.get((ntype, wname))
                if tpl is None:
                    inputs[wname] = wval  # no template available -> leave literal
                    continue
                tword = tpl["type"] if tpl["type"] != "dropdown" else "dropdown"
                pid = f"comfyrawworkflowinput{tword}{ntype.lower()}node{wname.replace('_','')}{chr(97 + nid)}"
                group = {"name": f"{ntype} (Node {nid})", "id": ntype.lower(),
                         "open": False, "priority": 0, "advanced": True,
                         "can_shrink": True, "toggles": False}
                if tpl["type"] == "decimal":
                    inputs[wname] = num_ph(pid, _fmt(wval))
                else:
                    inputs[wname] = txt_ph(pid, str(wval))
                add_param(tpl, pid, wval, tword, group=group)

        prompt[str(nid)] = {"inputs": inputs, "class_type": ntype, "_meta": {"title": ntype}}

    return {
        "workflow": enrich_workflow(g, props),
        "prompt": prompt,
        "custom_params": custom_params,
        "param_values": param_values,
        "image": "/imgs/model_placeholder.jpg",
        "description": description,
        "enable_in_simple": False,
    }


def convert(builder_module, out_name, description=""):
    std_tpl, auto_tpl, props = load_templates()
    g = builder_module.build()
    builder_module.validate(g)
    env = to_envelope(g, std_tpl, auto_tpl, props, description)
    out = SWARM_CW / out_name
    out.write_text(json.dumps(env, indent=2))
    n_params = len(env["custom_params"])
    print(f"wrote {out_name}  ({len(g['nodes'])} nodes, {n_params} params) -> {SWARM_CW.name}/")
    return env


if __name__ == "__main__":
    convert(build_m2, "Face-Hashing_M2.json",
            "M2 inpaint (fp16): keep the photo's scene/pose, swap face to the HASHED identity.")
    convert(build_m2_portrait, "Face-Hashing_M2_portrait.json",
            "M2 portrait baseline (fp16, 512): empty key = identity passthrough; set key to hash.")
