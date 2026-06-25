# Flux.1-dev + PuLID — identity pivot (RunPod)

Flux.2 Klein's PuLID port couldn't carry identity even at full strength on the isolated portrait.
**Flux.1-dev + lldacing's PuLID-Flux is the combo with a real identity track record** — this doc is
the same isolation test on it, to get a clean read on whether a convincing likeness is achievable at
Flux quality. If it is, we graduate this to the inpaint + bootstrap; if it still isn't, the problem is
upstream of the model (face detection / reference) and no model swap fixes it.

- **Workflow:** [`workflows/FLUX1_pulid_portrait.json`](workflows/FLUX1_pulid_portrait.json) (from
  [`build_flux1_pulid.py`](build_flux1_pulid.py)). Portrait/txt2img, denoise 1.0 — **zero photo
  leakage**, so the face you get is purely what PuLID reconstructs.
- **Shared pod setup** (Blackwell sm_120 torch, venv-aware pip, `ml_dtypes>=0.5.1`, the `HF_TOKEN`
  gated-download recipe): see [`FLUX2_PULID_GETTING_STARTED.md`](FLUX2_PULID_GETTING_STARTED.md)
  §0–§2 — all of it applies here unchanged.

```
LoadImage(face crop) → ApplyPulidFlux ─────────────┐  (antelopev2 ArcFace + EVA-CLIP, weight ≤ 5.0)
UNETLoader(flux1-dev) → ApplyPulidFlux → KSampler ◄─┘   EmptySD3 latent · cfg 1 · FluxGuidance 4 · denoise 1.0
```

---

## 1 — The node (you probably already have it)

Your earlier startup log listed **`comfyui_pulid_flux_ll` as IMPORT FAILED — on the exact ml_dtypes
error we already fixed**. That *is* this node (lldacing/ComfyUI_PuLID_Flux_ll). So after the
`ml_dtypes>=0.5.1` fix + a restart it should import. Verify: search the node menu for `Apply PuLID`
→ **`ApplyPulidFlux`** (category `pulid`). If it's there, skip to §2.

If it's missing or still erroring, (re)install into ComfyUI's interpreter (`$PYBIN` from the
FLUX2 doc §1):

```bash
cd $COMFY/custom_nodes
[ -d ComfyUI_PuLID_Flux_ll ] || git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git
"$PYBIN" -m pip install -r ComfyUI_PuLID_Flux_ll/requirements.txt
"$PYBIN" -m pip install facenet-pytorch --no-deps      # per the node's README
```

> Node names don't collide with the Flux.2 PuLID node (`ApplyPuLIDFlux2` vs `ApplyPulidFlux`), so
> both can coexist. Requires ComfyUI ≥ 0.3.7 (you're well past that).

---

## 2 — Weights

Only **flux1-dev (bf16) is gated** — accept its license and use the same `HF_TOKEN` curl recipe as
the FLUX2 doc §2. Everything else is open.

| File | `$COMFY/models/…` | Source | Gated? |
|---|---|---|---|
| `flux1-dev.safetensors` (bf16) | `diffusion_models/` | `black-forest-labs/FLUX.1-dev` | 🔒 yes |
| `t5xxl_fp8_e4m3fn.safetensors` | `text_encoders/` | `comfyanonymous/flux_text_encoders` | no |
| `clip_l.safetensors` | `text_encoders/` | `comfyanonymous/flux_text_encoders` | no |
| `ae.safetensors` (Flux VAE) | `vae/` | `black-forest-labs/FLUX.1-schnell` | no |
| `pulid_flux_v0.9.1.safetensors` | `pulid/` | `guozinan/PuLID` | no |
| `EVA02_CLIP_L_336_psz14_s6B.pt` | `clip/` | `QuanSun/EVA-CLIP` | no |
| antelopev2 `*.onnx` | `insightface/models/antelopev2/` | (already on the volume from InstantID/Klein) | no |

```bash
HF=https://huggingface.co
# gated — needs HF_TOKEN with the FLUX.1-dev license accepted (see FLUX2 doc §2)
curl -fL -H "Authorization: Bearer $HF_TOKEN" -o $COMFY/models/diffusion_models/flux1-dev.safetensors \
  $HF/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
# open
curl -fL -o $COMFY/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors $HF/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
curl -fL -o $COMFY/models/text_encoders/clip_l.safetensors           $HF/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
curl -fL -o $COMFY/models/vae/ae.safetensors                         $HF/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
curl -fL -o $COMFY/models/pulid/pulid_flux_v0.9.1.safetensors        $HF/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors
curl -fL -o $COMFY/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt         $HF/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt
```

> **VRAM (32 GB):** flux1-dev bf16 (~22 GB) + PuLID; ComfyUI frees the T5 encoder before sampling,
> so t5xxl-fp8 keeps peak comfortable. Want zero gating / lighter? Use `Comfy-Org/flux1-dev`'s
> **`flux1-dev-fp8.safetensors`** instead — but that's an all-in-one checkpoint, so swap
> `UNETLoader`+`DualCLIPLoader`+`VAELoader` for a single `CheckpointLoaderSimple`. Identity is
> slightly better on bf16, which is why the default targets it.

---

## 3 — Run the portrait test

1. Open `FLUX1_pulid_portrait.json` (restart ComfyUI first).
2. **`LoadImage`** → a **tight, frontal, well-lit crop of just your face** (~512–1024 px). This is
   the single biggest identity lever — a small/angled face gives a weak embedding.
3. **`PulidFluxInsightFaceLoader`** provider = `CUDA` (fall back to `CPU` if onnxruntime has no sm_120).

| Node | Setting | Default | Notes |
|---|---|---|---|
| `ApplyPulidFlux` | `weight` | **1.0** | range to **5.0** — push **1.5–2.0** if identity is soft. More headroom than Klein. |
| `ApplyPulidFlux` | `start_at` / `end_at` | 0.0 / 1.0 | apply across the whole denoise (keep for max identity) |
| `FluxGuidance` | `guidance` | **4.0** | the PuLID-Flux recipe |
| `KSampler` | steps / cfg | **20 / 1.0** | flux1-dev needs ~20 (cfg stays 1; guidance is the FluxGuidance node) |
| `KSampler` | sampler / scheduler | `euler` / `simple` | |

---

## 4 — Read the result

- **Looks like you →** identity *is* achievable here. Reply and I'll build the **inpaint** variant
  (keep scene, swap face — high denoise + feathered mask, no pixel-leak crutch) and wire flux1-dev +
  this node into `pod_bootstrap.sh`.
- **Still not you →** the problem is upstream of the model. Drop in **`PulidFluxFaceDetector`**
  (inputs: `face_analysis`, `image`, a `PulidFluxOptions`; outputs `embed_face` / `align_face` /
  `face_bbox_image`) → `PreviewImage` to *see* exactly which face antelopev2 detected and how it
  aligned it. If that crop isn't a clean shot of you, fix the reference / `PulidFluxOptions`
  (`input_faces_index`) before blaming the model.

---

## 5 — Where the hash plugs in (later)

`pulidflux.py`, `ApplyPulidFlux.apply_pulid_flux()`:

```python
face_info = face_analysis.get(image[i])
iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0)   # ← 512-d antelopev2 ArcFace
...
id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)            # EVA-CLIP concatenated here
```

Same plan as the Klein node: a `FaceHashApplyPulidFlux` subclass hashing `face_info.embedding` by
key before it becomes `iface_embeds` (antelopev2 → share the key with `FaceHashDepth`). **Same
EVA-CLIP leak caveat** — `id_cond_vit` still describes the original face, so hashing only the ArcFace
branch is a partial scramble. Decide how to handle that once we've confirmed identity works at all.
