# Face Hashing — Project Context

## What I'm building

Face Hashing is a personal, exploratory project.

- **In:** a photograph of a person (eventually video / multiple people).
- **Out:** the same photograph, but the person's face is different — and *deterministically* so: the same input face always yields the same output face. Under the hood the transform should behave like a hash (deterministic, ideally one-way — in practice *obfuscation*, not encryption).

The original four-stage picture:

1. A face model extracts what makes a face unique (the "facial features as JSON").
2. A math transform mutates those values (not their structure) into a *different* face — a hot-swappable "hash function".
3. Reconstruct a rough mesh from the transformed values (a Skyrim-character-creator-like draft).
4. A diffusion model makes it photoreal and composites it back into the original image.

## Where I am right now (current: 2026-06)

**Stages 1 & 3 run end-to-end, locally on the Mac** (Apple Silicon, no GPU, no Colab). One command:

```bash
cd local && python run.py -i in -o out      # out/<stem>/{arcface.npy, composed.glb}
```

- **Stage 1 — identity (MICA).** Photo → frozen ArcFace 512-d embedding → MICA → 300-d FLAME *shape* (identity). `arcface.npy` is the recognition vector (the Stage-2 key); the 300-d shape drives the mesh. Identity consistency on a small set: ArcFace AUC ~1.0 (ceiling) > MICA ~0.94 > DECA ~0.86 — MICA sits just under the ceiling, which is why it replaced DECA for identity.
- **Stage 3 — reconstruct (SMIRK).** SMIRK reads the photo's expression/pose/jaw/eyelids; we swap MICA's shape into SMIRK's FLAME and decode → `composed.glb`: the identity wearing the photo's expression. SMIRK (CVPR'24, MIT) replaced DECA here for much better expressions.
- **Stage 2 — the hash — is NOT built.** Today `composed.glb` uses the *raw* MICA shape (a faithful reconstruction). Stage 2 will mutate that 300-d shape vector before compose. `viewer/sliders.html` previews this: 300 live FLAME shape sliders + a deterministic, key-seeded "hash" offset.
- **Stage 4 — photoreal + composite — is NOT built.** The only stage that may need a GPU (diffusion behind an API); everything else is local.

## Architecture (local, renderer-free)

The key insight: MICA / SMIRK / DECA inference paths are **PyTorch3D-free** — only their *Renderers* need it. So we import each model's encoder + FLAME decoder, skip the Renderer, and there's nothing to build on macOS.

- `local/run.py` — the one command. MICA and SMIRK are separate upstream repos that **both** ship top-level `utils`/`configs`/`datasets` packages, so they can't co-exist in one interpreter; `run.py` runs each as a subprocess, drops intermediates in a temp dir, and assembles `out/<stem>/{arcface.npy, composed.glb}`.
- `local/mica_local.py` — MICA: `load` → `embed` / `arcface_embed` / `reconstruct` / `compose_mesh`. CPU matches the original Colab demo at cosine 1.0.
- `local/smirk_local.py` — SMIRK: `load` → `params` / `compose` / `reconstruct`. Mediapipe detector with an antelopev2/RetinaFace fallback for faces mediapipe misses.
- `local/compose_flame.py` — alternate compose (MICA shape + DECA params); legacy, kept for when DECA params exist.
- `local/patch_mica_for_mac.py` — idempotent MICA-on-Mac patcher.
- `local/README.md` — the runbook.
- `tools/present.py` — inspect/visualize identity + arcface (distance heatmap, PCA scatter, fingerprint).
- `tools/export_flame_basis.py` — export the FLAME shape basis for `viewer/sliders.html`.
- `viewer/index.html` — `<model-viewer>` page (drag-rotate a `.glb`).
- `viewer/sliders.html` — interactive 300-slider FLAME viewer with live hash.

## Repo layout

```
face-hashing/
├── CLAUDE.md                       # guidance for Claude Code
├── CONTEXT.md                      # this file — current truth
├── face_hashing_research_report.md # deep tool reference for all four stages
├── stage2-design.md                # Stage-2 plan: identity key file, two modes, local-first topology
├── local/                          # the Mac-native pipeline (the real workflow)
│   ├── run.py                      # one command: photos -> {arcface.npy, composed.glb}
│   ├── mica_local.py               # Stage 1: MICA identity (300-d shape) + ArcFace (512-d)
│   ├── smirk_local.py              # Stage 3: SMIRK expression/pose + compose with MICA shape
│   ├── compose_flame.py            # alternate compose (MICA shape + DECA params) — legacy
│   ├── patch_mica_for_mac.py       # makes a MICA checkout run on CPU/MPS (idempotent)
│   └── README.md                   # the runbook
├── tools/
│   ├── present.py                  # inspect/visualize identity + arcface
│   └── export_flame_basis.py       # export FLAME basis for the slider viewer
├── viewer/
│   ├── index.html                  # <model-viewer> page
│   ├── sliders.html                # interactive FLAME shape sliders + live hash
│   └── models/.gitkeep             # .glb files go here (git-ignored)
└── .gitignore
```
(Personal photos `local/in/`, model checkouts `local/MICA` + `local/smirk`, outputs `local/out*`, the exported `viewer/flame/` basis, and generated `.glb`/figures are git-ignored.)

## Mac footguns (the ones that bit us)

- **Python 3.11 arm64**, not 3.14 — chumpy has no wheels there; install with `pip install --no-build-isolation chumpy` (its setup.py imports `pip`).
- **`torch.load`** — checkpoints were saved from CUDA; force `map_location="cpu"`, and `weights_only=False` (torch ≥2.6 flipped the default) for the non-tensor payloads. Both runners patch this globally.
- **MPS** has no float64 — cast tensors to float32 before `.to(device)`.
- **numpy 2.0** removed aliases (`np.float`, `np.float_`, `np.bool`, `np.Inf`, …) that FLAME's pickles and chumpy reference — restore them before importing the model source.
- **`mica.testing = True`** — skips MICA `decode()`'s training-only `codedict['flame']` ground-truth block (the identity path is unchanged).
- **SMIRK detector** — mediapipe's FaceLandmarker misses small-in-frame / profile faces; we fall back to MICA's antelopev2/RetinaFace to locate the face, then re-run mediapipe on a padded crop (native crop framing preserved).
- **Viewer** — `<model-viewer>` can't load `.glb` over `file://`; serve over HTTP. The exported FLAME basis under `viewer/flame/` is license-gated — keep it local, don't commit or serve publicly.

## Design decisions

- **MICA over DECA for identity** — DECA `shape` drifts photo-to-photo (a reconstruction objective, not an identity one); MICA returns a pose/expression-invariant identity (the consistency numbers above).
- **SMIRK over DECA for expression** — better expressions, plus jaw + eyelids (CVPR'24, MIT).
- **Renderer-free** — params + mesh come from the encoder + FLAME decoder; no rasterizer, nothing to build on macOS.
- **Two subprocesses, not one process** — the `utils`/`configs`/`datasets` name collision between MICA and SMIRK makes a single interpreter fragile; isolate them.
- **Two identity vectors** — ArcFace 512-d is the identity/key; the MICA 300-d shape is the payload (a lossy projection of the 512-d). See `stage2-design.md`.
- **Untextured coarse mesh** (gray vertex color), Skyrim aesthetic; FLAME verts + faces → `.glb` via trimesh (`<model-viewer>` doesn't load `.obj`).

## Open questions / next

- **Stage 2:** implement `transform(shape_300, key) -> shape'` as a hot-swappable strategy (start: seeded Gaussian offset, clamp ±σ; preserve expression/pose). Mutate before compose. See `stage2-design.md`.
- **Stage 4:** diffusion (InstantID / PuLID / Arc2Face) for photoreal + composite — the one stage that may live behind a GPU API.

## Reference

- `local/README.md` — the runbook (setup, weights, verification, running the pipeline).
- `face_hashing_research_report.md` — deep tool reference for all four stages.
- `stage2-design.md` — Stage-2 plan: encrypted identity key file, two modes, keyed transform, local-first topology.
