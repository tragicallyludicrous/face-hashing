# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Face Hashing** is a personal, exploratory project. Goal: take a photo of a person and return the same photo with a *different* face — where the transform behaves like a hash (deterministic: same input face → same output face; ideally one-way). It is a research/prototyping repo, not a product. There is no build system, package manager, or test suite.

It runs **entirely locally on the Mac** (Apple Silicon, no NVIDIA GPU) — no Colab. The model inference paths we use are PyTorch3D-free (only their *renderers* need it), so there's nothing to build: we import each model's encoder + FLAME decoder and skip the renderer.

The eventual pipeline is **four stages** (see `face_hashing_research_report.md` for the deep dive on tooling at each stage):

1. **Extract** — face → structured representation. **MICA** turns the photo into a 300-d FLAME *shape* (identity) code, via a frozen ArcFace 512-d embedding. (DECA was the earlier pick; MICA's identity is pose/expression-invariant and far more consistent photo-to-photo — AUC ~0.94 vs DECA ~0.86, against an ArcFace ceiling of ~1.0.)
2. **Transform** — the "hash function": `transform(shape, key) -> shape'`. **Not implemented yet.** Hot-swappable strategy (e.g. `flame_shape_offset_v1`); mutate the MICA `shape` (identity), preserve expression/pose.
3. **Reconstruct** — a composed FLAME mesh: MICA shape (identity) + **SMIRK** expression/pose/jaw/eyelids from the original photo, decoded through SMIRK's FLAME. (SMIRK replaced DECA for expression — CVPR'24, MIT, much better.)
4. **Photorealism + composite** — diffusion model (InstantID / PuLID / Arc2Face) to make it a photo and paste back into the original image. Not built; the only stage that might need a GPU behind an API.

> Note on "one-way": true cryptographic one-wayness is not achievable here — treat the transform as *obfuscation*, not encryption (see report Key Finding 3 and the RiDDLE paper for the closest reversible-with-key prior art).

## Current state — Stages 1 & 3 local, end to end

One command turns a folder of photos into per-person identity + posed mesh:

```bash
cd local && python run.py -i in -o out      # out/<stem>/{arcface.npy, composed.glb}
```

`arcface.npy` is the 512-d ArcFace identity (the Stage-2 key / recognition vector); `composed.glb` is the MICA identity wearing the photo's SMIRK expression/pose. The composed identity is currently the **raw** MICA shape (a faithful reconstruction) — Stage 2 will mutate that shape vector (the hash) before compose.

`local/README.md` is the runbook (setup, weights, verification). `CONTEXT.md` is the "where am I right now" file — read it first.

## Local architecture (Mac-native, renderer-free)

There is no single packaged app; it's a thin driver over two model runners plus a static viewer:

- **`local/run.py`** — the one command. Orchestrates the two runners as **subprocesses** (MICA and SMIRK are separate upstream repos that both ship top-level `utils`/`configs`/`datasets` packages, so they can't share one interpreter), writes intermediates to a temp dir, and assembles only `arcface.npy` + `composed.glb` into `out/<stem>/`.
- **`local/mica_local.py`** — MICA in-process: `load` → `embed` (300-d shape) / `arcface_embed` (512-d) / `reconstruct` / `compose_mesh`. CPU output matches the original Colab demo at cosine 1.0; MPS works.
- **`local/smirk_local.py`** — SMIRK in-process (the Stage-3 sibling): `load` → `params` / `compose` / `reconstruct`. `compose` swaps MICA's shape into SMIRK's FLAME so its jaw/eyelids apply natively. Detector: mediapipe first, falling back to MICA's antelopev2/RetinaFace for small-in-frame / profile faces mediapipe misses.
- **`local/patch_mica_for_mac.py`** — idempotent patcher that makes a MICA checkout run on CPU/MPS (chumpy shim, numpy-2.0 aliases, `LandmarksType._2D`→`TWO_D`, CUDA→CPU provider, CPU-mapped `torch.load`).
- **Viewer (the Mac, in a browser).** `viewer/index.html` drag-rotates a `.glb` via Google's `<model-viewer>`; `viewer/sliders.html` is an interactive 300-slider FLAME shape viewer with a live deterministic hash preview, built over a locally-exported FLAME basis (`tools/export_flame_basis.py` → `viewer/flame/`).

## Commands

**Run the pipeline (local):**

```bash
cd local && python run.py -i in -o out        # photos -> out/<stem>/{arcface.npy, composed.glb}
# --device cpu|mps|auto   --detector antelopev2|vision   --keep-work
```

Individual stages also run standalone: `python mica_local.py -i in -o out/MICA` and `python smirk_local.py -i in -o out/SMIRK --mica out/MICA`. See `local/README.md`.

**Serve a viewer locally (on the Mac):**

```bash
cd viewer && python3 -m http.server 8080      # then open http://localhost:8080/ (index.html or sliders.html)
```

`<model-viewer>` cannot load `.glb` over `file://`; it must be served over HTTP.

**Inspect identity/embeddings:** `python tools/present.py --compare --source mica|arcface`.

## Mac setup notes

Full setup is in `local/README.md`. The non-obvious bits: use **Python 3.11 arm64** (not 3.14 — chumpy has no wheels), `pip install --no-build-isolation chumpy`, MPS has no float64 (cast to float32), and numpy 2.0 removed aliases FLAME's pickles need. The MICA and SMIRK checkouts live under `local/` (git-ignored); weights are downloaded per each repo's instructions. `mica.testing = True` is set so MICA's `decode()` skips a training-only ground-truth block (identity path unchanged).

## Local-only / license-gated files (git-ignored)

`.gitignore` keeps these out of the repo — keep it that way:

- `local/in/` — **personal photos** of real people. Local only.
- `local/out*/`, `local/MICA/`, `local/smirk/` — generated outputs and the (research-only-weight) model checkouts.
- `viewer/flame/` — the FLAME shape basis exported for `sliders.html`. **Registration-gated, non-commercial — do not commit it, and serve the viewer only locally** (committing or serving it publicly would redistribute FLAME weights).
- `viewer/models/*.glb` — generated meshes (derive from personal photos).
- `tools/figures/` — generated inspection figures (carry real-person labels).
- `Drive Folder` — machine-specific macOS bookmark (legacy Colab handoff).

Licensing: **FLAME 2020** is registration-gated, non-commercial (CC-BY with content restrictions). **MICA** and **DECA** pretrained weights are research-only. **SMIRK** is MIT. **EMOCA** is registration-required. None of this is relicensed by running locally — a shipped app would be license-gated.

## Doc map

- `CONTEXT.md` — **current truth**: live state, the renderer-free / local-first decisions, design history. Start here.
- `local/README.md` — the **runbook**: Mac setup, weights, the cross-check, running `run.py` / `mica_local` / `smirk_local`.
- `face_hashing_research_report.md` — deep reference on tools/approaches/pricing for all four stages. Consult when designing Stage 2+.
- `stage2-design.md` — the **Stage-2 plan**: the encrypted per-user identity key file, the two run modes (recurring vs one-off), enrollment, the keyed transform registry, and the local-first topology. Design, not yet built.
