# Face Hashing: A Practical Build Guide for a Deterministic, Hot-Swappable Face-to-Synthetic-Face Pipeline

## TL;DR
- **Build the MVP on an API stack: ArcFace (via InsightFace) → a deterministic transform on the 512-dim embedding → Arc2Face or PuLID/InstantID via Replicate or fal.ai for image synthesis → InsightFace `inswapper` (or InstantID's IdentityNet) to composite back into the original photo.** This delivers your four-stage architecture with the least code and roughly $0.002–$0.07 per generated image.
- **The closest match to your "Skyrim character generator" mental model is a 3DMM pipeline (DECA / EMOCA / MICA → FLAME parameters)** — you get a literal JSON-like dict of shape, expression, pose, and detail coefficients. It is heavier to set up and gives lower photorealism than identity-embedding diffusion; use it only if semantic interpretability matters more than image quality.
- **True cryptographic one-wayness is *not* achievable here**: face anonymization is an active research area (CIAGAN, DeepPrivacy2, FALCO, RiDDLE, FLUID, NullFace) and an adversary with the same transform and a face database can often re-identify. Your transform should be treated as *obfuscation*, not encryption — design it as a clean `transform(features, key) -> features'` strategy interface so you can iterate.

---

## Key Findings

1. **Stage 1 (extraction) has three fundamentally different output types**: dense opaque embeddings (ArcFace 512-d, FaceNet 128/512-d), geometric landmarks (MediaPipe's 468 3D points + 52 ARKit-style blendshapes), and semantic 3DMM parameters (FLAME shape/expression/pose). Your "facial features as JSON" mental model maps most cleanly onto FLAME parameters from DECA/EMOCA/MICA. ArcFace embeddings are simpler to obtain but opaque — every coordinate is a learned dimension, not a nameable feature.

2. **Stage 2 (transform) is where the "hash" lives.** For embeddings, use seeded orthogonal rotations or HMAC-derived offsets on the 512-d vector — fast, deterministic, and stays roughly on the unit hypersphere ArcFace embeddings occupy. For FLAME parameters, perturb each coefficient deterministically by a key-derived offset clamped to ±2σ of the prior. Keep this as a single Python callable behind a strategy interface so the algorithm is hot-swappable.

3. **Stage 3 (rough reconstruction) is optional.** Arc2Face collapses Stage 3 and Stage 4 into one model: feed it an ArcFace embedding, get a photorealistic face. If you do want an intermediate "rough draft," DECA/FLAME renders a 3D mesh from parameters via PyTorch3D, and e4e/pSp/ReStyle invert into StyleGAN's W+ latent for a 1024×1024 face.

4. **Stage 4 (photorealistic composite) is dominated by three SOTA tools as of 2026**: InstantID (most balanced quality/speed, requires SDXL + IdentityNet ControlNet), PuLID/PuLID-Flux (highest identity fidelity, FLUX or SDXL backbone), and IP-Adapter FaceID/FaceID-PlusV2 (lightest, broadest model compatibility). For pure face-region replacement preserving the rest of the image, InsightFace's `inswapper_128` remains the de-facto open-source baseline.

5. **API pricing is cheap enough for prototyping.** Per the Replicate model pages: `zsxkib/instant-id` is approximately $0.025 per run on an L40S (about 40 runs per $1); `bytedance/flux-pulid` is approximately $0.029 per run on an A100 80GB (about 34 runs per $1, quoted directly from replicate.com/bytedance/flux-pulid); `bytedance/pulid` (SDXL classic) is approximately $0.0020 per run on L40S (about 500 runs/$1); `tencentarc/photomaker` is approximately $0.0069 per run on L40S (about 144 runs/$1). On fal.ai, `fal-ai/flux-pulid` is "$0.0333 per megapixel". AWS Rekognition is $1.00 per 1,000 images for IndexFaces/DetectFaces (first 1M tier) plus $0.01 per 1,000 face vectors per month for storage.

6. **Azure Face API is gated.** Per Microsoft Learn: "Face service access is limited based on eligibility and usage criteria in order to support our Responsible AI principles. Face service is only available to Microsoft managed customers and partners," and use by/for US police departments is explicitly prohibited. Plan around Azure unless you have already cleared the intake form.

7. **Local hardware floor.** 12 GB VRAM (RTX 3060 12GB or better) for SDXL + IP-Adapter FaceID at 1024×1024; 16 GB recommended once you add ControlNet/InstantID; 24 GB (RTX 4090) for FLUX-based PuLID with headroom. RunPod's RTX 4090 is $0.69/hr on secure cloud and $0.34/hr on community cloud (confirmed by RunPod's own GPU page, titled "RTX 4090 GPU Cloud | $0.69/hr GPUs on-demand," and corroborated by SynpixCloud's April 2026 snapshot: "RunPod RTX 4090 price per hour is $0.34 for community cloud and $0.69 for secure cloud as of April 2026").

---

## Details

### STAGE 1 — Face recognition / embedding / landmark extraction

The first decision is *what kind of representation* you want to mutate. There are three families.

**A. Dense identity embeddings (recommended starting point).** A single 128–512-dim vector trained to be discriminative between identities.

- **InsightFace** (`pip install insightface onnxruntime-gpu`, https://github.com/deepinsight/insightface). The `buffalo_l` model pack bundles RetinaFace detection, 5-point landmarks, and an ArcFace iresnet-100 backbone producing a **512-dim L2-normalized embedding**. This is the industry default; `app.get(np.asarray(img))[0].embedding` returns a `(512,)` numpy array. License: code is MIT but model weights and the `inswapper_128` swap model are non-commercial; commercial use requires a license from insightface.ai.
- **DeepFace** (https://github.com/serengil/deepface) — Python wrapper that lets you swap between VGG-Face, FaceNet (128/512-d), OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet, and Buffalo_L with a single `model_name=` argument. It also bundles `DeepFace.analyze()` which returns age, gender, emotion, and race. Per the official serengil/deepface README: "Age model got ± 4.65 MAE; gender model got 97.44% accuracy, 96.29% precision and 95.05% recall as mentioned in its tutorial." This is useful if you want a *semantic* JSON-style feature dict alongside (or instead of) an embedding.
- **FaceNet / ArcFace standalone** — `pip install arcface`, or `garavv/arcface-onnx` on Hugging Face, gives you the raw 512-d vector without InsightFace's license strings.
- **face_recognition / dlib** — 128-d embedding (`face_recognition.face_encodings(image)`), CPU-friendly, lowest accuracy. Fine for prototyping; upgrade later.

**B. Geometric landmarks.** A list of (x, y, z) points.

- **MediaPipe Face Mesh / Face Landmarker** (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) — returns **468 3D landmarks** (or 478 with `refine_landmarks=True` to include iris). The newer Face Landmarker also emits **52 ARKit-style blendshape coefficients** (jawOpen, eyeBlinkLeft, browInnerUp, …) and a transformation matrix. Runs on CPU/mobile in real time. Available in Python (`mediapipe`) and JavaScript (`@mediapipe/tasks-vision`).
- **dlib** — Classic 68-point 2D landmarks; older but stable.

**C. 3D Morphable Model (3DMM) parameters — the closest fit to your "JSON object" intuition.** A small set of *semantically meaningful* coefficients that parameterize a 3D face mesh.

- **FLAME** (https://flame.is.tue.mpg.de/) — the dominant modern 3DMM. Parameters split into ~300 **shape** coefficients (identity sliders), ~100 **expression** blendshape coefficients, **pose** (jaw rotation, neck, global rotation), and **appearance/texture**. Requires registration (non-commercial license).
- **DECA** (https://github.com/yfeng95/DECA, SIGGRAPH 2021) — feed-forward regressor: image → FLAME shape+expression+pose+detail-displacement parameters. The output is literally a Python dict / `.npy` file with named tensors — this *is* the structured JSON-style representation you described.
- **EMOCA** (https://emoca.is.tue.mpg.de/, CVPR 2022) — DECA plus an emotion-consistency loss for better expression capture.
- **MICA** (https://github.com/Zielon/MICA, ECCV 2022) — explicitly metrical (real-world millimeter scale) FLAME identity reconstruction.
- **INFERNO / EMICA** (https://github.com/radekd91/inferno) — combines DECA + EMOCA + SPECTRE + MICA into the current best-in-class FLAME regressor.
- **Pixel3DMM** (arXiv 2505.00615, 2025, https://simongiebenhain.github.io/pixel3dmm/) — newer optimization-based single-image 3D face reconstruction.

**APIs (cloud).**

- **AWS Rekognition** — `IndexFaces` extracts a feature vector and stores it server-side in a collection; you do *not* get the raw embedding back. You get a `FaceId` UUID, bounding box, and (with `DetectFaces`) attributes like age range, smile, eyeglasses, emotions, head pose. **Pricing: $1.00 per 1,000 images for IndexFaces/DetectFaces in the first 1M-image tier, plus $0.01 per 1,000 face vectors per month for face metadata storage** (aws.amazon.com/rekognition/pricing/). Best for face search/verification, not for getting an embedding you can mutate.
- **Azure Face API** — Limited Access. From Microsoft Learn: "Face service access is limited based on eligibility and usage criteria in order to support our Responsible AI principles. Face service is only available to Microsoft managed customers and partners." Emotion and gender attributes have been **retired**; age, smile, facial hair, hair, and makeup are **limited**. Skip Azure unless you have an enterprise account that already cleared the intake form.
- **Google Cloud Vision FACE_DETECTION** — Returns bounding box, 34 landmarks, head pose, and likelihood scores for joy/sorrow/anger/surprise/under-exposed/blurred/headwear. Pricing per cloud.google.com/vision/pricing: "first 1,000 units of image processing per month [free]. For usage beyond this, it charges $1.50 per 1,000 units up to 5 million units, with the rate dropping to $0.60 per 1,000 units thereafter." No raw embedding output.
- **Face++** (Megvii) — Landmarks, attributes, and a comparison endpoint. Geographic/regulatory friction; not recommended unless you specifically need it.

**Recommendation for Stage 1**: Start with **InsightFace `buffalo_l`** locally — it's free, fast, gives you a real 512-d vector you can transform, and is the same model that feeds InstantID/PuLID/IP-Adapter FaceID under the hood (those tools use the `antelopev2` pack, which is closely related). If you want the semantic JSON, layer **DECA** (or INFERNO/EMICA) on top to also extract FLAME parameters. Combining both — ArcFace embedding for identity + FLAME parameters for geometry/expression — is the **hybrid architecture** that gives you the most expressive transform space.

### STAGE 2 — Deterministic transforms (the "hash function")

This is the core IP of your project. The transform is `f(features, key) -> features'` and must satisfy: deterministic (same input → same output), structure-preserving (output lies on or near the manifold of plausible faces), visually distinct (output face is clearly different from input), and ideally hard to invert.

**For dense embeddings (e.g., 512-d ArcFace vector):**

ArcFace embeddings live on a hypersphere (L2-normalized; the angular margin loss explicitly trains them this way). A good transform respects that geometry:

- **Seeded orthogonal rotation.** Generate a random rotation matrix `R ∈ SO(512)` from your key (QR decomposition of a key-seeded Gaussian matrix). Apply `e' = R @ e`. Norm-preserving, deterministic, and fully invertible *if* the attacker has the key — which is precisely the "encryption-like" property. To make it less invertible, compose with a non-linear step.
- **Hash-then-project.** `e' = normalize(e + α · project_to_tangent(H(key, e)))` where `H` is a keyed PRF (HMAC-SHA256 expanded to 512 floats via SHAKE). Adds key-dependent drift while staying near the original direction; α controls "how different."
- **Seeded permutation + sign flip.** Permute the 512 indices and negate a key-determined half. Trivially deterministic; surprisingly disruptive to identity while preserving the vector's statistical properties.
- **Latent walk in a learned identity space.** Project into StyleGAN W+ (via e4e/pSp/ReStyle) or the Arc2Face conditioning space and apply a small deterministic offset along discovered identity directions (InterFaceGAN-style boundaries for age/gender/identity).
- **Quantization + lookup.** Quantize the embedding to a hash bucket, deterministically map bucket → synthetic identity from a fixed pre-generated gallery. Loses granularity but guarantees stability against tiny pose/lighting changes in the input — *exactly* the "same person from a different photo gives the same hash" property you may want.

**Critical detail.** ArcFace embeddings should be L2-normalized after the transform before feeding into Arc2Face / InstantID / PuLID, all of which expect unit-norm ArcFace-distributed vectors.

**For FLAME / 3DMM parameters:**

The parameter vector is structured (shape: ~300-d, expression: ~100-d, pose: ~6-d). Treat each block separately:

- **Shape (identity)** — mutate the most. Add a key-derived offset within ±2σ of the FLAME shape prior (approximately Gaussian per coefficient). Clamping keeps faces plausible.
- **Expression** — usually preserve (so the output face inherits the same smile).
- **Pose** — preserve (so the output face looks the same direction).
- **Detail / displacement** — optionally re-synthesize from a key-derived seed for skin-texture variation.

This gives a very clean "Skyrim slider" experience: literal JSON object `{"shape": [...300 floats...], "expression": [...100 floats...], "pose": [...]}` and your hash function is `shape += seeded_offset(key)`.

**The "hot-swap" pattern.**

Define a minimal interface and dependency-inject the transform. A `HashTransform` protocol with `(features: dict, key: bytes) -> dict`, then register implementations: `rotation_v1`, `hmac_perm_v1`, `flame_shape_offset_v1`, `hybrid_arc_flame_v1`. Pick one per request. Strategy pattern; lets you A/B-test transforms against the same input set and the same downstream generator.

**Why true one-wayness is hard.** Active research:

- **CIAGAN** (CVPR 2020, https://github.com/dvl-tum/ciagan) — conditional GAN that removes identity while preserving pose; identity is controlled by a discriminator, not a hash.
- **DeepPrivacy / DeepPrivacy2** (https://github.com/hukkelas/deep_privacy2) — inpaints faces with synthetic ones; the generator never sees the original face.
- **FALCO** (CVPR 2023 highlight, https://github.com/chi0tzp/FALCO) — optimizes in StyleGAN2's W+ latent to push identity a controlled distance away while preserving attributes via FaRL feature loss.
- **RiDDLE** (CVPR 2023) — *reversible* anonymization with a key — the closest paper to your "cryptographic" framing. Encrypts identity into the latent and can decrypt with the right key. **If you want a real reversibility-with-key property, study this paper.**
- **FLUID** (arXiv 2511.17005, Nov 2025) — training-free latent-space identity substitution via diffusion model h-space.
- **NullFace** (arXiv 2503.08478, 2025) — training-free localized face anonymization with diffusion.

The literature consistently shows that **attribute-preserving anonymization leaks identity**: an attacker with the same anonymizer + a face DB can often link back. Treat your hash as obfuscation, not encryption.

### STAGE 3 — Reconstruct a rough face from the transformed representation

Optional, depending on your Stage 4 choice.

**If you went the 3DMM route (FLAME parameters):**

- Use FLAME's PyTorch layer (https://github.com/soubhiksanyal/FLAME_PyTorch) to convert parameters → 5023-vertex mesh.
- Render with **PyTorch3D**, **nvdiffrast**, or DECA's built-in rasterizer. This gives you the "Skyrim character preview" — a recognizable but obviously CG face mesh with proper geometry/expression/pose.
- Optionally apply FLAME's albedo model (AlbedoMM) for a texture pass.

**If you went the embedding route:**

- **Arc2Face** (https://github.com/foivospar/Arc2Face, https://huggingface.co/FoivosPar/Arc2Face) — exactly the tool for this. Input: a 512-d ArcFace embedding. Output: a photorealistic 512×512 face image. Per the paper (arXiv:2403.11641): "Arc2Face builds upon a pretrained Stable Diffusion model … we meticulously upsample a significant portion of the WebFace42M database, the largest public dataset for face recognition (FR)" — i.e., it trained on an upsampled subset of WebFace42M, not the full dataset; Arc2Face was accepted to ECCV 2024 as an oral presentation. There is also an Arc2Face + FLAME blendshape IP-Adapter for joint identity + expression control. This is the **single best match for your architecture** because it literally takes the ArcFace vector you transformed in Stage 2 and emits the face image.
- **StyleGAN inversion encoders** — `pSp` (https://github.com/eladrich/pixel2style2pixel), `e4e` (https://github.com/omertov/encoder4editing), `ReStyle` (https://github.com/yuval-alaluf/restyle-encoder). Project an image into W+ (18×512 = 9,216-d), manipulate, decode through FFHQ-trained StyleGAN2/3 at 1024×1024. Useful if you want classic GAN latent editing (age/smile/pose via InterFaceGAN directions) but produces lower ID fidelity than ArcFace-conditioned diffusion.

### STAGE 4 — Photorealistic generation + composite back into the original photo

Two sub-tasks: (a) generate a high-quality face from the transformed representation, (b) put it back into the source image with matching pose/expression/lighting.

**Identity-conditioned diffusion (generate from transformed identity):**

- **InstantID** (https://github.com/InstantID/InstantID) — SDXL + an "IdentityNet" ControlNet + an IP-Adapter projection layer. Takes an ArcFace embedding (via `antelopev2`) plus a pose-keypoint image. Best balance of quality/speed in the 2026 PuLID-vs-InstantID-vs-FaceID community comparisons. Replicate: `zsxkib/instant-id` at ~$0.025/run (~40 runs/$1) on L40S. Documented at https://replicate.com/docs/guides/make-images-of-real-people-instantly-with-instant-id.
- **PuLID / PuLID-Flux** (https://github.com/ToTheBeginning/PuLID) — newer (mid-2024), contrastive-alignment based, higher identity fidelity. SDXL and FLUX backbones. fal.ai: `fal-ai/flux-pulid` at "$0.0333 per megapixel" (fal.ai/models/fal-ai/flux-pulid). Replicate: `bytedance/flux-pulid` at "$0.029 per run on Replicate, or 34 runs per $1" on A100 80GB (replicate.com/bytedance/flux-pulid); `bytedance/pulid` (SDXL classic) at ~$0.0020/run (~500 runs/$1) on L40S; `fofr/pulid-base` at ~$0.013/run on L40S.
- **IP-Adapter FaceID / FaceID-Plus / FaceID-PlusV2** (https://huggingface.co/h94/IP-Adapter-FaceID) — drop-in IP-Adapter using ArcFace embeddings (Plus variant also fuses CLIP) with optional LoRA for ID consistency. Lightest, broadest compatibility (works with any SD1.5/SDXL checkpoint). 8 GB VRAM works; 12 GB ideal. Lowest ID fidelity of the three but most "iteration-friendly."
- **PhotoMaker / PhotoMaker V2** (https://github.com/TencentARC/PhotoMaker, CVPR 2024) — "stacked ID embedding" from 1–4 reference images. Replicate: `tencentarc/photomaker` at ~$0.0069/run (~144 runs/$1) on L40S.
- **Arc2Face** — see Stage 3; can serve as both Stage 3 and Stage 4 if you do not need to preserve the original photo's background.

**Classic face-swap (preserve everything but the face region):**

- **InsightFace `inswapper_128`** (the model behind Roop / FaceFusion / ReActor) — non-commercial license. Takes a source identity face + a target image; outputs the target image with the face replaced. 128×128 output, typically post-processed with **GPEN** or **CodeFormer** to 512 or 1024. Community consensus: per InsightFace's own site, this is "the de-facto standard for open-source face swapping, delivering the best quality results." For commercial use, contact insightface.ai for licensing (or pay for the newer `inswapper-512-live` / Picsi.ai models).
- **SimSwap** (https://github.com/neuralchen/SimSwap) — 256 and unofficial 512 versions. Better preservation of target's hair/jawline/lighting; lower ID fidelity than inswapper.
- **FaceShifter, GHOST, BlendSwap, UniFace, FastFake** — academic alternatives; see the 1337sheets comparison and the September 2025 preprints.org watermarking study for honest side-by-side evaluations.
- **DeepFaceLab / Rope** — heavy per-pair training; overkill for hashing.

**Inpainting-based hybrid:**

Mask the face region (use the InsightFace bounding box or the MediaPipe Face Mesh contour as the mask). Run **SDXL inpainting** or **FLUX Fill** conditioned on the transformed identity (via InstantID or IP-Adapter FaceID with the inpainting checkpoint). This composes "generate *this* person, in *this* pose, *here*." Wei Mao's "FaceDetailer + InstantID + IP-Adapter" ComfyUI workflow on OpenArt is a battle-tested template for this.

**Pose / expression / lighting matching:**

- InstantID's IdentityNet conditions on a keypoint image, so passing the original photo's face keypoints (from InsightFace) makes the generated face inherit pose automatically.
- For tighter expression control: feed FLAME expression blendshape parameters into the Arc2Face + FLAME blendshape IP-Adapter, or use a ControlNet variant that conditions on a rendered mesh.
- For lighting: post-process with a relighting model (e.g., **IC-Light**, **DiffusionRig**) or rely on the SDXL inpainting context to match.

**ComfyUI workflow recipe.**

ComfyUI is the right environment for the "hot-swap experimentation" phase because each node is a swappable function.

A reasonable graph:
1. `Load Image` → `InsightFace Detector` → bounding box + 5-point landmarks + ArcFace embedding.
2. **Custom Python node** → your `HashTransform.apply(embedding, key)`.
3. `Apply InstantID` (IdentityNet ControlNet + IP-Adapter, with the transformed embedding injected in place of the reference image's embedding) → generated full face image.
4. `FaceDetailer` (inpainting on the face mask) or `inswapper` → composite.
5. `Upscale` (GPEN / CodeFormer).

For embedding injection, you will need the `ComfyUI_IPAdapter_plus` (cubiq) or the `ComfyUI_InstantID` nodes — they normally take a reference image, but you can patch them to accept a pre-computed embedding tensor.

**Pricing summary (per generated image, 2026):**

| Service | Model | Approximate price |
|---|---|---|
| Replicate | `zsxkib/instant-id` | $0.025 (~40 runs/$1), L40S |
| Replicate | `bytedance/pulid` (SDXL) | $0.0020 (~500 runs/$1), L40S |
| Replicate | `bytedance/flux-pulid` | $0.029 (~34 runs/$1), A100 80GB |
| Replicate | `tencentarc/photomaker` | $0.0069 (~144 runs/$1), L40S |
| Replicate | `fofr/face-to-many` | $0.0087 (~114 runs/$1), L40S |
| fal.ai | `fal-ai/flux-pulid` | $0.0333 per megapixel |
| fal.ai | `fal-ai/photomaker` | ~$0.001 per compute-second |
| AWS Rekognition | IndexFaces / DetectFaces | $1.00 per 1,000 images (first 1M tier) |
| AWS Rekognition | Face metadata storage | $0.01 per 1,000 vectors/month |
| Google Cloud Vision | FACE_DETECTION | First 1,000/mo free; $1.50 per 1,000 to 5M; $0.60 per 1,000 above |
| RunPod | RTX 4090 secure cloud | $0.69/hr (community cloud $0.34/hr) |
| RunPod | A100 80GB | $1.90/hr secure |

For prototyping at a few hundred test images, the API stack costs well under $10.

### Local hardware requirements

| Workflow | Minimum VRAM | Comfortable | Notes |
|---|---|---|---|
| InsightFace ArcFace | — | — | Runs on CPU |
| MediaPipe / DECA | 4 GB | 8 GB | Pre-trained; no training needed |
| SDXL base, 1024×1024 | 8 GB | 12 GB | `--medvram` needed at 8 GB |
| SDXL + IP-Adapter FaceID | 8 GB | 12 GB | Lightest identity-conditioned option |
| SDXL + InstantID (ControlNet + IP-Adapter) | 12 GB | 16 GB | RTX 3060 12GB borderline |
| FLUX + PuLID | 16 GB (Q5/Q6 GGUF) | 24 GB FP8 | FP16 needs more |
| Arc2Face (SD1.5 base) | 8 GB | 12 GB | |
| StyleGAN2/3 FFHQ inversion (e4e, pSp) | 8 GB | 12 GB | |

### Python and JS ecosystem

**Python wrappers:**
- `diffusers` (Hugging Face) — InstantID, IP-Adapter FaceID, PhotoMaker, PuLID all have pipelines.
- `insightface` — ArcFace, RetinaFace, inswapper.
- `deepface` — high-level wrapper for ArcFace/FaceNet/etc. + attributes.
- `mediapipe` — face mesh, blendshapes.
- `replicate`, `fal-client` — API clients.
- `comfyui` + custom-node ecosystem — visual prototype layer.

**JS / web frontend:**
- `@mediapipe/tasks-vision` — MediaPipe Face Mesh / Face Landmarker in the browser via WebGPU / WASM.
- `face-api.js` (TensorFlow.js) — landmarks + 128-d FaceNet embedding client-side, useful for live preview.
- `replicate` (Node) and `@fal-ai/serverless-client` — call your backend.
- For UI: Next.js + React + `react-dropzone` for upload, MediaPipe for live face-mesh preview, then submit to a Python backend (FastAPI) that runs ComfyUI's API or talks to Replicate/fal.

---

## Recommendations

### Minimum viable prototype (build this first, ~1 weekend)

1. **Backend (Python, FastAPI):**
   - `POST /hash-face` accepts an image + a `key` + a `transform_name`.
   - Run InsightFace `buffalo_l` to detect face, get bounding box + 512-d embedding.
   - Apply the selected `HashTransform` (start with seeded orthogonal rotation + L2 renormalize).
   - Two-call pipeline option for simplest path:
     1. Render the transformed embedding through **Arc2Face** locally (or on Replicate) → a synthetic reference image of the "new identity."
     2. Call Replicate `zsxkib/instant-id` (or run InstantID locally) with that synthetic image as `face_image` and the original photo's face-keypoint image as the pose condition.
   - Optionally finish with InsightFace `inswapper` to tighten the face-region paste-back.
   - Return the final image.

2. **Frontend (Next.js):**
   - Drag-and-drop upload, transform dropdown ("rotation_v1", "permutation_v1", "flame_shape_offset_v1", "hybrid_v1"), seed/key input, side-by-side original-vs-hashed display.

3. **Tests:**
   - **Determinism:** same image + same key → byte-identical output (within JPEG noise).
   - **Cross-photo consistency:** two different photos of the same person + same key → outputs that look like the *same* synthetic person (hashed embeddings should be close).
   - **Distinctness:** same image + different keys → visually different synthetic people.

### Staged path forward

| Stage | Trigger to move on | Next move |
|---|---|---|
| MVP (Replicate + ArcFace + InstantID) | API works end-to-end, < $10 spent | Add a 3DMM branch (DECA) to also extract FLAME params |
| Hybrid features | You can demo a "Skyrim slider" UI on FLAME shape coeffs | Add an Arc2Face-direct path (skip Stage 4 swap) |
| Local migration | API costs > $50/mo or you want offline | Stand up ComfyUI on RTX 3060 12GB or RunPod RTX 4090 ($0.69/hr secure, $0.34/hr community) |
| Identity-robustness | Same person, different photos → different hashes | Quantize embeddings to coarse buckets / use multi-image average (cf. AWS user vectors) |
| Research-grade reversibility | You want a "decrypt with key" property | Re-implement on top of RiDDLE (CVPR 2023) |

### Concrete library install list

```
pip install insightface onnxruntime-gpu mediapipe deepface
pip install diffusers transformers accelerate
pip install replicate fal-client
pip install fastapi uvicorn pillow numpy
pip install xformers          # if running diffusion locally
git clone https://github.com/yfeng95/DECA          # FLAME params
git clone https://github.com/foivospar/Arc2Face   # embedding → image
```

### Thresholds that should change the recommendation

- **If identity-preservation across photos matters more than "different-face" output** → switch from random rotation to quantization-into-K-buckets + fixed gallery of K pre-generated synthetic faces.
- **If you need > 1k images/day** → move from Replicate to local ComfyUI on a rented RTX 4090.
- **If you ever ship a commercial product** → InsightFace's `inswapper` and `buffalo_l` weights are non-commercial; budget for the InsightFace commercial license or switch to a fully open-license stack (SimSwap + custom-trained ArcFace).
- **If reversibility-with-key becomes a requirement** → study and re-implement RiDDLE; don't try to make plain orthogonal rotation "secure."

---

## Caveats

- **This is obfuscation, not encryption.** Face anonymization is an active research area precisely because no current method gives strict cryptographic one-wayness while keeping the output a plausible face. Don't ship this as a privacy guarantee for anyone whose face actually matters.
- **InsightFace model licenses are non-commercial.** The `buffalo_l` recognition pack, the `inswapper_128` swap model, and the `antelopev2` pack used by InstantID/PuLID are all academic/non-commercial. A commercial product needs explicit licensing.
- **Azure Face is gated.** Don't plan around it unless you've already passed the Limited Access intake form.
- **InstantID, PuLID, and IP-Adapter all internally use the same ArcFace family.** The embedding you transform in Stage 2 is the same embedding they consume — convenient, but also means an attacker who knows your transform and has ArcFace can attempt inversion. Composing a non-linear, non-invertible step into your transform is essential if you care about inversion resistance.
- **Determinism is fragile across face-detector versions.** ArcFace embeddings depend on the face crop / alignment; if you upgrade the detector (RetinaFace → SCRFD → newer), embeddings shift slightly and your hash output drifts. Pin model versions, and ideally bucketize the embedding before transforming.
- **Photorealism on the face boundary is the hardest part.** Even SOTA `inswapper` + GPEN leaves a faint seam under harsh lighting; budget time for the compositing step, not just the generation step.
- **Some pricing numbers fluctuate.** Replicate per-run prices vary with input image size and inference time; the figures quoted are the model pages' "approximately $X to run" headlines as of mid-2026 and are not contractual.
- **`fal-ai/face-swap` and `fal-ai/instant-id` may not currently be first-party fal endpoints** — `fal-ai/flux-pulid` and `fal-ai/photomaker` are confirmed; for InstantID specifically, Replicate is the more reliable host.
- **The "lucataco/pulid" Replicate model does not exist** — the canonical PuLID authors on Replicate are `bytedance/pulid`, `bytedance/flux-pulid`, and `fofr/pulid-base`. Use those identifiers.