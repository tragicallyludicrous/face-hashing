"""Builder for M2_instantid_hashed.json — the Stage-2+4 milestone workflow.

M2 = keep the photo's scene + pose, swap the face to the HASHED identity.

  LoadImage(photo) ─┬─ image (identity → hashed by FaceHashApplyInstantID)
                    ├─ image_kps (pose/expression, untouched)
                    └─ VAEEncode → SetLatentNoiseMask(face_mask) → KSampler
  InstantID model + antelopev2 + InstantID ControlNet drive the identity;
  the hash (key) makes that identity a deterministic *different* person.

Emits a litegraph workflow JSON. Run:  python comfy/build_m2.py
"""

import json
import pathlib

# (type, widgets, [in (name,type)], [out (name,type)], pos)
NODES = [
    ("CheckpointLoaderSimple", ["RealVisXL_V5.0_fp32.safetensors"], [],
     [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], [40, 40]),
    ("CLIPTextEncode", ["RAW photo, a person, natural skin texture, same lighting and background, photorealistic, high detail"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 40]),
    ("CLIPTextEncode", ["cartoon, cgi, 3d render, illustration, painting, drawing, blurry, deformed, disfigured, plastic skin"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 240]),
    ("LoadImage", ["zack-normal-06.jpeg", "image"], [],
     [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 440]),
    ("LoadImageMask", ["zack-normal-06_mask.png", "red"], [], [("MASK", "MASK")], [40, 700]),
    ("InstantIDModelLoader", ["ip-adapter.bin"], [], [("INSTANTID", "INSTANTID")], [40, 880]),
    ("InstantIDFaceAnalysis", ["CPU"], [], [("FACEANALYSIS", "FACEANALYSIS")], [40, 1000]),
    ("ControlNetLoader", ["instantid-controlnet-sdxl.safetensors"], [],
     [("CONTROL_NET", "CONTROL_NET")], [40, 1120]),
    ("FaceHashApplyInstantID",
     ["zacks-secret", 0.0, 0.8, 0.8, 0.0, 1.0, 0.0, "average"],
     [("instantid", "INSTANTID"), ("insightface", "FACEANALYSIS"), ("control_net", "CONTROL_NET"),
      ("image", "IMAGE"), ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
      ("image_kps", "IMAGE"), ("mask", "MASK")],
     [("MODEL", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")], [720, 440]),
    ("VAEEncode", [], [("pixels", "IMAGE"), ("vae", "VAE")], [("LATENT", "LATENT")], [720, 120]),
    ("SetLatentNoiseMask", [], [("samples", "LATENT"), ("mask", "MASK")], [("LATENT", "LATENT")], [1050, 120]),
    ("KSampler", [42, "fixed", 30, 4.5, "dpmpp_2m", "karras", 0.9],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
     [("LATENT", "LATENT")], [1320, 440]),
    ("VAEDecode", [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [1660, 440]),
    ("SaveImage", ["facehash_M2"], [("images", "IMAGE")], [], [1660, 620]),
]

# connections by node index: (src_node, src_out_slot, dst_node, dst_in_slot)
CK, POS, NEG, IMG, MSK, IID, FA, CN, FH, VE, SLN, KS, VD, SAVE = range(14)
CONNS = [
    (CK, 0, FH, 4), (CK, 1, POS, 0), (CK, 1, NEG, 0), (CK, 2, VE, 1), (CK, 2, VD, 1),
    (POS, 0, FH, 5), (NEG, 0, FH, 6),
    (IMG, 0, FH, 3), (IMG, 0, FH, 7), (IMG, 0, VE, 0),
    (MSK, 0, FH, 8), (MSK, 0, SLN, 1),
    (IID, 0, FH, 0), (FA, 0, FH, 1), (CN, 0, FH, 2),
    (FH, 0, KS, 0), (FH, 1, KS, 1), (FH, 2, KS, 2),
    (VE, 0, SLN, 0), (SLN, 0, KS, 3),
    (KS, 0, VD, 0), (VD, 0, SAVE, 0),
]


def build():
    nodes, links = [], []
    # scaffold node dicts
    for i, (ntype, widgets, ins, outs, pos) in enumerate(NODES):
        nodes.append({
            "id": i + 1, "type": ntype, "pos": pos, "size": [300, 120],
            "flags": {}, "order": i, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in ins],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": s}
                        for s, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": ntype},
            "widgets_values": list(widgets),
        })
    # wire links
    for lid, (sn, so, dn, di) in enumerate(CONNS, start=1):
        ltype = NODES[sn][3][so][1]
        links.append([lid, sn + 1, so, dn + 1, di, ltype])
        nodes[dn]["inputs"][di]["link"] = lid
        nodes[sn]["outputs"][so]["links"].append(lid)

    return {
        "last_node_id": len(NODES), "last_link_id": len(CONNS),
        "nodes": nodes, "links": links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
    }


def validate(g):
    by_id = {n["id"]: n for n in g["nodes"]}
    for lid, sn, so, dn, di, lt in g["links"]:
        assert so < len(by_id[sn]["outputs"]), f"link {lid}: bad out slot"
        assert di < len(by_id[dn]["inputs"]), f"link {lid}: bad in slot"
        assert by_id[dn]["inputs"][di]["link"] == lid, f"link {lid}: input not wired"
        assert lid in by_id[sn]["outputs"][so]["links"], f"link {lid}: output not wired"
    # every input slot must be connected (this graph has no optional-unconnected inputs)
    for n in g["nodes"]:
        for inp in n["inputs"]:
            assert inp["link"] is not None, f"{n['type']} input '{inp['name']}' unconnected"
    return True


if __name__ == "__main__":
    g = build()
    validate(g)
    out = pathlib.Path(__file__).with_name("M2_instantid_hashed.json")
    out.write_text(json.dumps(g, indent=2))
    print(f"wrote {out}  ({len(g['nodes'])} nodes, {len(g['links'])} links) — validated")
