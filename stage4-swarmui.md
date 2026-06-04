# Stage 4 — SwarmUI (ComfyUI backend) photoreal composite: setup manual

**Goal:** take a transformed FLAME mesh (the hashed identity, posed to the photo) and paint a
photorealistic face of that new identity back into the original photo — keeping hair, body,
background, and lighting. This is the runbook for standing that up in **SwarmUI** on the M2 Max.
Meant to be **resumed over several sessions** — see the [progress tracker](#progress-tracker) at the
bottom and fill in the "where I left off" notes as you go.

> **SwarmUI is ComfyUI with a nicer front-end.** Its generation **backend *is* ComfyUI**; SwarmUI adds
> a parameter-driven **Generate** tab, a **Models** browser, and embeds the full node editor as the
> **Comfy Workflow** tab. So every engine concept here (nodes, models, ControlNet, MPS behavior) is
> identical to ComfyUI — you just install/manage through SwarmUI, and drop into the Comfy Workflow tab
> for our custom graph. If the macOS SwarmUI build ever gets janky (it's more tested on Win/Linux,
> and needs .NET), raw ComfyUI is a drop-in fallback — the rest of this doc still applies.

> The ecosystem moves fast. Treat exact filenames/paths below as "true as of writing — verify in
> SwarmUI's Models tab / on Hugging Face." When a node name has drifted, search it in the Comfy
> Workflow tab's manager rather than trusting this doc.

Background: `face_hashing_research_report.md` §"STAGE 4" (tool survey + pricing). Architecture recap
in `CONTEXT.md` / `CLAUDE.md`. The conditioning images come from our **bridge** (`local/render_cond.py`).

---

## 0. The picture (what we're wiring)

Stage 4 = **three jobs**, each a signal the diffusion model consumes:

| Job | Signal we feed it | Where it comes from |
|---|---|---|
| **Structure / pose** | a depth (and/or normal) map of the mesh, aligned to the photo's camera | the **bridge**: render the composed/hashed mesh → PNGs (`render_cond.py`) |
| **Identity** | the geometry itself (Path A) and/or an ArcFace embedding / reference face (Path B) | the mesh; `arcface.npy`; or Arc2Face / a frontal render of the hashed mesh |
| **Preservation** | an inpaint **mask** (only face/head/neck is regenerated) + a lighting pass | the bridge's mesh-silhouette mask; IC-Light |

Two ways in, build them in order:

- **Path A — geometry-driven (start here).** mesh → depth/normal → **ControlNet** → SDXL **inpaint**
  the masked face. Identity carried by geometry; needs nothing but the mesh. Most direct proof.
- **Path B — identity-driven (add later).** **InstantID** conditioned on an ArcFace of the *new*
  face + the photo's pose. Stronger recognizable identity; where `arcface.npy` pays off.

End state is the hybrid: ControlNet-depth (pose/shape) + InstantID (identity) + inpaint mask
(preserve) + IC-Light (lighting).

---

## 1. Hardware & expectations (read first)

- **M2 Max, 96 GB unified.** Capacity is a non-issue (SDXL + ControlNets + IP-Adapter fit easily).
  **Speed is the constraint** — the ComfyUI backend runs on **MPS**, far slower than CUDA: budget
  **~30 s–2 min/image for SDXL**, **several min for Flux-PuLID**. Iterate at 768–1024 px, 20–30 steps.
- **Start on SDXL, not Flux.** Mature Mac support; Flux on MPS is slow and node support thinner.
- **xformers does not exist on Apple Silicon** — never install it. The backend auto-uses split/
  sub-quad attention.
- **API escape hatch:** when MPS is too slow or a node won't run, point a step at Replicate/fal
  (see §8). Good pattern: local SwarmUI for geometry/inpaint experiments, API for Flux-PuLID finals.

---

## 2. Run SwarmUI (you already have it installed)

- Launch it (the macOS launch script in your SwarmUI folder, e.g. `./launch-macos.sh`, or however
  you start it). It needs **.NET 8**; the heavy compute is its **ComfyUI backend** (Python/MPS),
  which SwarmUI installs/manages for you under `dlbackend/`.
- Open the web UI (SwarmUI default: **http://localhost:7801**).
- **Confirm the backend is healthy:** *Server → Backends* should show the ComfyUI backend as
  connected/green. If it's not, that's the first thing to fix (see §9).
- Generate a quick test image from the **Generate** tab (any checkpoint) to confirm MPS works end to
  end before adding anything.

Two tabs you'll live in:
- **Generate** — parameter-driven (prompt, model, ControlNet image, LoRAs). Great for M0 and quick
  iteration.
- **Comfy Workflow** — the embedded full ComfyUI node editor. This is where our M1–M3 graphs live.
  You can also "Import" a Comfy `.json` here, or push a Generate-tab setup into it as a starting point.

---

## 3. Where things go (models + custom nodes in SwarmUI)

**Models** live under **`SwarmUI/Models/<subdir>/`** and show up in the Models tab (which can also
download for you). SwarmUI maps these into the ComfyUI backend automatically:

| What | SwarmUI folder |
|---|---|
| SDXL checkpoints | `Models/Stable-Diffusion/` |
| ControlNet models | `Models/controlnet/` |
| CLIP-Vision | `Models/clip_vision/` |
| LoRAs | `Models/Lora/` |
| VAE | `Models/VAE/` |

**Custom nodes** (InstantID, etc.) install into the **backend ComfyUI's `custom_nodes/`** — find it
under your SwarmUI install (e.g. `SwarmUI/dlbackend/.../ComfyUI/custom_nodes/`); the *Server* tab
shows the backend path. `git clone` the node repos there (or use the Comfy-Manager from inside the
Comfy Workflow tab if present), then **restart the backend** (Server → Backends → restart).

**InstantID's insightface path is special:** the InstantID node reads ArcFace from the *backend
ComfyUI's* `models/insightface/models/antelopev2/` (not SwarmUI/Models). We already have that pack —
copy it straight in:

```bash
# point at YOUR backend ComfyUI dir (check Server → Backends for the exact path):
BACKEND=~/SwarmUI/dlbackend/.../ComfyUI          # <-- edit this
mkdir -p "$BACKEND/models/insightface/models"
cp -R ~/.insightface/models/antelopev2 "$BACKEND/models/insightface/models/"
ls "$BACKEND/models/insightface/models/antelopev2"   # expect 5 .onnx
```

It's the same ArcFace our `mica_local`/`smirk_local` use — same recognition model end to end.

---

## 4. Custom nodes to install

Install into the backend's `custom_nodes/` (§3). What each is for:

| Node pack | Repo | Why |
|---|---|---|
| **ComfyUI_InstantID** | cubiq/ComfyUI_InstantID | Path B identity (ArcFace + IdentityNet ControlNet). |
| **ComfyUI_IPAdapter_plus** | cubiq/ComfyUI_IPAdapter_plus | IP-Adapter FaceID/Plus — lighter identity + style. |
| **comfyui_controlnet_aux** | Fannovel16/comfyui_controlnet_aux | ControlNet preprocessors (we render our own depth, but handy). |
| **ComfyUI-Impact-Pack** | ltdrdata/ComfyUI-Impact-Pack | **FaceDetailer** + mask ops. |
| **ComfyUI-IC-Light** | kijai/ComfyUI-IC-Light | Relighting pass (SD1.5-based — see §6). |

Restart the backend after installing.

---

## 5. Models to download

Via SwarmUI's Models tab (preferred) or place files by hand into the §3 folders.

| What | File(s) (as of writing) | SwarmUI folder | Source |
|---|---|---|---|
| **SDXL photoreal checkpoint** | `RealVisXL_V5.0.safetensors` (or `Juggernaut-XL`) | `Models/Stable-Diffusion/` | Civitai / HF |
| **SDXL ControlNet — depth** | depth-sdxl `diffusion_pytorch_model.safetensors` | `Models/controlnet/` | HF `xinsir/controlnet-depth-sdxl-1.0` |
| **SDXL ControlNet — normal** (opt.) | normal-sdxl controlnet | `Models/controlnet/` | HF |
| **InstantID — IP-Adapter** | `ip-adapter.bin` | backend `models/instantid/` | HF `InstantX/InstantID` |
| **InstantID — ControlNet** | `ControlNetModel/diffusion_pytorch_model.safetensors` | `Models/controlnet/` | HF `InstantX/InstantID` |
| **antelopev2 (ArcFace)** | 5 `.onnx` | backend `models/insightface/models/antelopev2/` | **copy from `~/.insightface` (§3)** |
| **IP-Adapter FaceID (SDXL)** | `ip-adapter-faceid_sdxl.bin` + LoRA | `Models/` (ipadapter) + `Models/Lora/` | HF `h94/IP-Adapter-FaceID` |
| **CLIP-Vision** | `CLIP-ViT-H-14` (and/or bigG) | `Models/clip_vision/` | HF |
| **IC-Light** (later) | `iclight_sd15_fc.safetensors` (+ SD1.5 base) | per node | HF `lllyasviel/ic-light` |

> InstantID/IPAdapter use folders the *backend* expects (`models/instantid`, `models/ipadapter`). If
> SwarmUI's Models tab doesn't surface them, drop the files into the backend ComfyUI's `models/<…>`
> directly. `onnxruntime` (CPU) is needed for insightface — comes with the InstantID node deps. Do
> **not** install `onnxruntime-gpu` on Mac.

---

## 6. The workflows (build in order)

Use the **Generate** tab for M0; build M1–M3 in the **Comfy Workflow** tab (same node graphs as
plain ComfyUI). Inputs from *our* pipeline come from the **bridge** + existing outputs.

### M0 — sanity (Generate tab)
Pick RealVisXL, prompt a portrait, generate. Confirms checkpoint + MPS produce a clean image. SwarmUI
also lets you add a ControlNet image right here — good for a first depth test before going to nodes.

### M1 — geometry inpaint (Path A; the real first test) — Comfy Workflow tab
Puts a *new* face into the photo using only the mesh geometry.

```
Load Checkpoint (RealVisXL) ─┬─ MODEL
                             ├─ CLIP → (pos)("photorealistic close-up portrait, natural skin, <scene/light>")
                             │        → (neg)("blurry, cgi, plastic, deformed, extra face")
                             └─ VAE
Load Image: original photo ── VAE Encode ── LATENT ── Set Latent Noise Mask ← MASK (face mask, from bridge)
Load Image: mesh DEPTH (from bridge) ┐
ControlNetLoader(depth-sdxl) ────────┤
ControlNetApplyAdvanced(pos, neg, control_net, depth, strength 0.6–0.9) → pos', neg'
KSampler(MODEL, pos', neg', masked LATENT, denoise 0.8–1.0, FIXED seed) → VAE Decode → Save Image
```

Knobs: **ControlNet strength** (mesh authority; start 0.7), **denoise in mask** (1.0 = fully new
face), **mask feather** (soft edge → fewer seams). Face ignores the mesh → raise strength; looks
pasted → feather mask + a final low-denoise full-image pass.

### M2 — add InstantID identity (Path B)
Needs a **reference of the new face** (frontal render of the hashed mesh, or Arc2Face from the hashed
`arcface`).

```
InstantIDModelLoader(ip-adapter.bin) ─ INSTANTID
InstantIDFaceAnalysis(antelopev2) ─ FACE_ANALYSIS
ControlNetLoader(instantid-controlnet) ─ CONTROL_NET
Load Image: NEW-identity reference face ─ IMAGE
Load Image: original photo (pose keypoints) ─ IMAGE  (optional image_kps)
ApplyInstantID(INSTANTID, FACE_ANALYSIS, CONTROL_NET, ref IMAGE, MODEL, pos, neg, weight 0.8) → MODEL', pos', neg'
→ same KSampler / inpaint tail as M1 (you can stack the depth ControlNet too)
```

> Advanced: InstantID/IPAdapter nodes take a *reference image* by default but can be patched to accept
> a **precomputed ArcFace embedding** — that's how you'd feed a hashed `arcface.npy` directly. Do this
> only after the image-reference path works.

### M3 — preservation polish
- **Mask** = the bridge's mesh-silhouette (face/head/neck) → only that region changes.
- **FaceDetailer** (Impact Pack) for a final face refine.
- **IC-Light** relight pass on the composite (it's SD1.5 — run as a separate stage).
- A final **low-denoise (0.15–0.25) full-image img2img** to harmonize seams/color.

**Determinism:** Stage 2's hash makes the *mesh* deterministic; Stage 4 just needs a **fixed KSampler
seed** + fixed conditioning for "same input → same output."

---

## 7. Where our pipeline plugs in

- **Mesh:** `local/out/<stem>/<stem>_composed.glb`, or a **hashed** mesh/shape from `viewer/studio.html`.
- **Conditioning (depth / normal / mask):** the **bridge** (`local/render_cond.py`) renders these from
  the mesh using SMIRK's camera so they align to the photo. Load via `Load Image` (or drag onto the
  Generate tab's ControlNet/mask slots).
- **Original photo:** `local/in/<stem>.<ext>`.
- **Identity (Path B):** `local/out/<stem>/<stem>_arcface.npy` or a hashed ArcFace. Quick reference:
  Arc2Face on that embedding, or a frontal render of the hashed mesh.

---

## 8. API fallback (when MPS is too slow / a node won't run)

From the report (verify current prices on the model pages):

| Provider | Model | ~Cost |
|---|---|---|
| Replicate | `zsxkib/instant-id` (SDXL) | ~$0.025/run (~40/$1) |
| Replicate | `bytedance/pulid` (SDXL) | ~$0.0020/run (~500/$1) |
| Replicate | `bytedance/flux-pulid` | ~$0.029/run (~34/$1) |
| fal.ai | `fal-ai/flux-pulid` | ~$0.0333 / megapixel |

SwarmUI can also register **remote/Comfy API backends** if you want it to drive a cloud GPU. For pure
face-region swap, InsightFace `inswapper_128` is the open baseline. **Critical:** L2-normalize any
ArcFace embedding *after* the hash transform before feeding Arc2Face/InstantID/PuLID.

---

## 9. Troubleshooting

**SwarmUI-specific**
- **Backend won't start / red in Server → Backends** → check the backend log in the Server tab; usually
  a Python/torch install issue in `dlbackend/.../ComfyUI`. You can `source` that venv and run its
  `python main.py` directly to see the real error.
- **A custom node isn't showing in Comfy Workflow** → it went into the wrong `custom_nodes` (must be
  the *backend's*), or the backend wasn't restarted.
- **Model not in the dropdown** → wrong `Models/` subfolder, or hit *Refresh* in the Models tab.
- **.NET errors on launch** → install/repair .NET 8 (arm64).

**MPS (the ComfyUI backend)**
- **Black / NaN images** → VAE precision. Use `--fp32-vae` for the backend, or the `sdxl-vae-fp16-fix`
  VAE. (Backend launch flags: Server → Backends → edit the ComfyUI backend's start args.)
- **"op not implemented for MPS"** → ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is in the backend's env.
- **InstantID "insightface model not found"** → antelopev2 isn't at the backend's
  `models/insightface/models/antelopev2/` (note the nested `models/`).
- **Painfully slow** → 768–896 px, 20–25 steps, `dpmpp_2m`/`karras`; only go 1024+ for finals.
- **xformers errors** → you installed it in the backend; `pip uninstall xformers`.

---

## Progress tracker

Update as you go (the "where did I leave off" memory).

- [ ] **M0** SwarmUI up, backend green, Generate-tab sanity image renders on MPS
- [ ] antelopev2 copied into the backend's `models/insightface/models/antelopev2/`
- [ ] Custom nodes installed (InstantID, IPAdapter_plus, controlnet_aux, Impact-Pack) + backend restarted
- [ ] Models in place (SDXL checkpoint, depth controlnet, InstantID files)
- [ ] **M1** geometry-inpaint graph produces a new face in the photo (needs the bridge maps)
- [ ] **M2** InstantID identity control working
- [ ] **M3** mesh-silhouette mask + FaceDetailer + IC-Light polish
- [ ] decided local SwarmUI vs API for "production" renders

**Where I left off / notes:**

>
>

**Working config (fill in once it works — saves re-deriving):**

- SwarmUI backend ComfyUI path:
- backend launch args (MPS flags):
- checkpoint:
- depth controlnet strength / denoise / steps / sampler that looked good:
