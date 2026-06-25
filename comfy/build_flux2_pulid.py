"""Builder for FLUX2_pulid_roundtrip.json — the Flux.2 Klein + PuLID identity MVP.

The round-trip / null test: take a photo, extract its OWN ArcFace identity, and
re-render the masked face region from that identity (PuLID). No hash yet — the
output should look like the SAME person. That proves the Flux.2 identity pipe end
to end *before* Stage-2 turns the identity into a different person.

  LoadImage(photo) ─┬─ image  → ApplyPuLIDFlux2 (identity, antelopev2 ArcFace)
                    └─ pixels → VAEEncode → SetLatentNoiseMask(face_mask) → KSampler
  UNETLoader(Klein) → ApplyPuLIDFlux2 → KSampler (masked denoise keeps the scene)

Base graph + node names mirror iFayens/ComfyUI-PuLID-Flux2 workflows/PuLID_Flux2_(Klein).json.
Klein is distilled: 4 steps / cfg 1 / euler / simple at full denoise. For *inpaint*
(denoise < 1) a 4-step schedule is too coarse, so we default to 8 steps / denoise 0.85
(tune in the .md). Target: RunPod RTX PRO 4500 (32 GB, Blackwell) — provider CUDA.

Emits a litegraph workflow JSON. Run:  python comfy/build_flux2_pulid.py
"""

import json
import pathlib

# (type, widgets, [in (name,type)], [out (name,type)], pos)
NODES = [
    ("UNETLoader", ["flux-2-klein-9b-fp8.safetensors", "default"], [],
     [("MODEL", "MODEL")], [40, 40]),
    ("CLIPLoader", ["qwen_3_8b_fp8mixed.safetensors", "flux2", "default"], [],
     [("CLIP", "CLIP")], [40, 200]),
    ("VAELoader", ["flux2-vae.safetensors"], [], [("VAE", "VAE")], [40, 360]),
    ("CLIPTextEncode",
     ["a candid photo of a person, natural skin texture, same lighting and background, "
      "photorealistic, sharp focus, 85mm"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 200]),
    ("CLIPTextEncode", [""], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 380]),
    ("LoadImage", ["zack-normal-06.jpeg", "image"], [],
     [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 480]),
    ("LoadImageMask", ["zack-normal-06_mask.png", "red"], [], [("MASK", "MASK")], [40, 720]),
    ("PuLIDModelLoader", ["pulid_flux2_klein_v2.safetensors"], [],
     [("PULID_MODEL", "PULID_MODEL")], [40, 860]),
    ("PuLIDEVACLIPLoader", [], [], [("EVA_CLIP", "EVA_CLIP")], [40, 980]),
    ("PuLIDInsightFaceLoader", ["CUDA"], [], [("INSIGHTFACE", "INSIGHTFACE")], [40, 1100]),
    # strength 1.3 (1.4 if identity is weak), face_index 0, debug False.
    # The hash will slot in here later: a FaceHashApplyPuLIDFlux2 subclass that hashes
    # face.embedding by key before PuLID reads it (antelopev2 — share the key with FaceHashDepth).
    ("ApplyPuLIDFlux2", [1.3, 0, False],
     [("model", "MODEL"), ("pulid_model", "PULID_MODEL"), ("eva_clip", "EVA_CLIP"),
      ("face_analysis", "INSIGHTFACE"), ("image", "IMAGE")],
     [("MODEL", "MODEL")], [720, 480]),
    ("VAEEncode", [], [("pixels", "IMAGE"), ("vae", "VAE")], [("LATENT", "LATENT")], [720, 120]),
    ("SetLatentNoiseMask", [], [("samples", "LATENT"), ("mask", "MASK")],
     [("LATENT", "LATENT")], [1040, 120]),
    # Klein distilled: cfg 1, euler/simple. Inpaint wants more than 4 steps at denoise<1.
    ("KSampler", [42, "fixed", 8, 1.0, "euler", "simple", 0.85],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
      ("latent_image", "LATENT")],
     [("LATENT", "LATENT")], [1360, 480]),
    ("VAEDecode", [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [1680, 480]),
    ("SaveImage", ["facehash_flux2_pulid"], [("images", "IMAGE")], [], [1680, 660]),
]

# connections by node index: (src_node, src_out_slot, dst_node, dst_in_slot)
(UNET, CLIP, VAE, POS, NEG, IMG, MSK, PULID, EVA, FACE,
 APPLY, VE, SLN, KS, VD, SAVE) = range(16)
CONNS = [
    (UNET, 0, APPLY, 0), (PULID, 0, APPLY, 1), (EVA, 0, APPLY, 2),
    (FACE, 0, APPLY, 3), (IMG, 0, APPLY, 4),
    (CLIP, 0, POS, 0), (CLIP, 0, NEG, 0),
    (POS, 0, KS, 1), (NEG, 0, KS, 2),
    (IMG, 0, VE, 0), (VAE, 0, VE, 1),
    (MSK, 0, SLN, 1), (VE, 0, SLN, 0), (SLN, 0, KS, 3),
    (APPLY, 0, KS, 0),
    (KS, 0, VD, 0), (VAE, 0, VD, 1), (VD, 0, SAVE, 0),
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
    # Emit into comfy/workflows/ — pod_bootstrap.sh symlinks that dir as ComfyUI's canvas
    # workflows folder, so a migrated pod's `git pull` makes this show up in the sidebar.
    out = pathlib.Path(__file__).parent / "workflows" / "FLUX2_pulid_roundtrip.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    print(f"wrote {out}  ({len(g['nodes'])} nodes, {len(g['links'])} links) — validated")
