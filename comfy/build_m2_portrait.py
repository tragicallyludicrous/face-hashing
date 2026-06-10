"""Builder for M2_portrait_fp16.json — the fast identity-baseline workflow.

Portrait variant of M2: txt2img (EmptyLatentImage 1024x1024, fp16) instead of the
1550x1550 inpaint, so it's quicker and isolates the *face InstantID builds* with no
inpaint-blend confound. (1024 matters — InstantID identity is weak at 512.)

  LoadImage(photo) ─┬─ image      (identity → FaceHashApplyInstantID)
                    └─ image_kps  (pose/keypoints, untouched)
  EmptyLatentImage(512x512) ─ latent ─ KSampler (denoise 1.0, from scratch)

The hash is BYPASSED by default: key="" → the node returns the identity
unchanged (see nodes.py: `if ... or not key: return out`). This is the control
test — "put my ID right back in" — so the portrait should look like the real
person. Flip the `key` field to a real key (e.g. "zacks-secret") in SwarmUI and
the SAME workflow renders the hashed, different identity for side-by-side compare.

Emits a litegraph workflow JSON. Run:  python comfy/build_m2_portrait.py
"""

import json
import pathlib

# (type, widgets, [in (name,type)], [out (name,type)], pos)
NODES = [
    ("CheckpointLoaderSimple", ["RealVisXL_V50_fp16.safetensors"], [],
     [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], [40, 40]),
    ("CLIPTextEncode", ["RAW photo, a person, natural skin texture, photorealistic, high detail, softly lit studio background"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 40]),
    ("CLIPTextEncode", ["cartoon, cgi, 3d render, illustration, painting, drawing, blurry, deformed, disfigured, plastic skin"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 240]),
    ("LoadImage", ["zack-normal-06.jpeg", "image"], [],
     [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 440]),
    ("InstantIDModelLoader", ["ip-adapter.bin"], [], [("INSTANTID", "INSTANTID")], [40, 680]),
    ("InstantIDFaceAnalysis", ["CPU"], [], [("FACEANALYSIS", "FACEANALYSIS")], [40, 800]),
    ("ControlNetLoader", ["instantid-controlnet-sdxl.safetensors"], [],
     [("CONTROL_NET", "CONTROL_NET")], [40, 920]),
    # key="" → identity passthrough (baseline). Set a key in SwarmUI to hash.
    # ip_weight=0 → IP-Adapter off; identity rides entirely on the IdentityNet ControlNet
    # (cn_strength=1.0), which still consumes the (hashable) ArcFace embedding. noise=0.6 de-burns.
    ("FaceHashApplyInstantID",
     ["", 0.0, 0.0, 1.0, 0.0, 1.0, 0.6, "average"],
     [("instantid", "INSTANTID"), ("insightface", "FACEANALYSIS"), ("control_net", "CONTROL_NET"),
      ("image", "IMAGE"), ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
      ("image_kps", "IMAGE")],
     [("MODEL", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")], [720, 440]),
    ("EmptyLatentImage", [1024, 1024, 1], [], [("LATENT", "LATENT")], [720, 120]),
    ("KSampler", [42, "fixed", 30, 4.5, "dpmpp_2m", "karras", 1.0],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
     [("LATENT", "LATENT")], [1320, 440]),
    ("VAEDecode", [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [1660, 440]),
    ("SaveImage", ["facehash_M2_portrait"], [("images", "IMAGE")], [], [1660, 620]),
]

# connections by node index: (src_node, src_out_slot, dst_node, dst_in_slot)
CK, POS, NEG, IMG, IID, FA, CN, FH, EL, KS, VD, SAVE = range(12)
CONNS = [
    (CK, 0, FH, 4), (CK, 1, POS, 0), (CK, 1, NEG, 0), (CK, 2, VD, 1),
    (POS, 0, FH, 5), (NEG, 0, FH, 6),
    (IMG, 0, FH, 3), (IMG, 0, FH, 7),
    (IID, 0, FH, 0), (FA, 0, FH, 1), (CN, 0, FH, 2),
    (FH, 0, KS, 0), (FH, 1, KS, 1), (FH, 2, KS, 2),
    (EL, 0, KS, 3),
    (KS, 0, VD, 0), (VD, 0, SAVE, 0),
]


def build():
    nodes, links = [], []
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
    for n in g["nodes"]:
        for inp in n["inputs"]:
            assert inp["link"] is not None, f"{n['type']} input '{inp['name']}' unconnected"
    return True


if __name__ == "__main__":
    g = build()
    validate(g)
    out = pathlib.Path(__file__).with_name("M2_portrait_fp16.json")
    out.write_text(json.dumps(g, indent=2))
    print(f"wrote {out}  ({len(g['nodes'])} nodes, {len(g['links'])} links) — validated")
