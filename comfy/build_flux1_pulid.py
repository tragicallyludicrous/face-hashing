"""Builder for FLUX1_pulid_portrait.json — Flux.1-dev + PuLID identity isolation test.

The pivot from Flux.2 Klein: Klein's PuLID port couldn't carry identity even at full strength
on the isolated portrait. Flux.1-dev + lldacing's mature PuLID-Flux is the combo with a real
identity track record. This is the SAME isolation test (EmptySD3 latent, denoise 1.0, zero photo
leakage) so the result is a clean read on whether identity is achievable at Flux quality.

  LoadImage(face crop) → ApplyPulidFlux (antelopev2 ArcFace + EVA-CLIP, weight up to 5.0)
  UNETLoader(flux1-dev) → ApplyPulidFlux → KSampler(EmptySD3 latent, cfg 1, FluxGuidance 4) → save

Node names + signatures from lldacing/ComfyUI_PuLID_Flux_ll (ApplyPulidFlux: model, pulid_flux,
eva_clip, face_analysis, image, weight, start_at, end_at). Emits a litegraph workflow JSON.
Run:  python comfy/build_flux1_pulid.py
"""

import json
import pathlib

# (type, widgets, [in (name,type)], [out (name,type)], pos)
NODES = [
    ("UNETLoader", ["flux1-dev.safetensors", "default"], [], [("MODEL", "MODEL")], [40, 40]),
    ("DualCLIPLoader", ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux", "default"],
     [], [("CLIP", "CLIP")], [40, 200]),
    ("VAELoader", ["ae.safetensors"], [], [("VAE", "VAE")], [40, 360]),
    ("CLIPTextEncode",
     ["a candid head-and-shoulders portrait photo of a person, natural skin texture, soft daylight, "
      "sharp focus, 85mm"],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 200]),
    ("FluxGuidance", [4.0], [("conditioning", "CONDITIONING")], [("CONDITIONING", "CONDITIONING")], [680, 200]),
    ("CLIPTextEncode", [""], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], [360, 380]),
    ("LoadImage", ["zack-face-crop.png", "image"], [], [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 480]),
    ("PulidFluxModelLoader", ["pulid_flux_v0.9.1.safetensors"], [], [("PULIDFLUX", "PULIDFLUX")], [40, 720]),
    ("PulidFluxEvaClipLoader", [], [], [("EVA_CLIP", "EVA_CLIP")], [40, 840]),
    ("PulidFluxInsightFaceLoader", ["CUDA"], [], [("FACEANALYSIS", "FACEANALYSIS")], [40, 960]),
    # weight 1.0 (range to 5.0 — push toward 1.5–2.0 if identity is weak), start_at 0.0, end_at 1.0.
    # Hash slots in here later: subclass ApplyPulidFlux, hash face_info.embedding by key before it
    # becomes iface_embeds (antelopev2 — share the key with FaceHashDepth). EVA-CLIP leak still applies.
    ("ApplyPulidFlux", [1.0, 0.0, 1.0],
     [("model", "MODEL"), ("pulid_flux", "PULIDFLUX"), ("eva_clip", "EVA_CLIP"),
      ("face_analysis", "FACEANALYSIS"), ("image", "IMAGE")],
     [("MODEL", "MODEL")], [720, 480]),
    ("EmptySD3LatentImage", [1024, 1024, 1], [], [("LATENT", "LATENT")], [720, 120]),
    # flux1-dev: cfg 1 (guidance comes from FluxGuidance), ~20 steps, euler/simple, denoise 1.0 (portrait).
    ("KSampler", [42, "fixed", 20, 1.0, "euler", "simple", 1.0],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
      ("latent_image", "LATENT")],
     [("LATENT", "LATENT")], [1040, 480]),
    ("VAEDecode", [], [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [1360, 480]),
    ("SaveImage", ["facehash_flux1_pulid_portrait"], [("images", "IMAGE")], [], [1360, 660]),
]

(UNET, DCLIP, VAE, POS, FG, NEG, IMG, PMODEL, EVA, FACE,
 APPLY, LAT, KS, VD, SAVE) = range(15)
CONNS = [
    (UNET, 0, APPLY, 0), (PMODEL, 0, APPLY, 1), (EVA, 0, APPLY, 2),
    (FACE, 0, APPLY, 3), (IMG, 0, APPLY, 4),
    (DCLIP, 0, POS, 0), (DCLIP, 0, NEG, 0),
    (POS, 0, FG, 0), (FG, 0, KS, 1), (NEG, 0, KS, 2),
    (APPLY, 0, KS, 0), (LAT, 0, KS, 3),
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
    out = pathlib.Path(__file__).parent / "workflows" / "FLUX1_pulid_portrait.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    print(f"wrote {out}  ({len(g['nodes'])} nodes, {len(g['links'])} links) — validated")
