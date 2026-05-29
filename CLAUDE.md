# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Face Hashing** is a personal, exploratory project. Goal: take a photo of a person and return the same photo with a *different* face — where the transform behaves like a hash (deterministic: same input face → same output face; ideally one-way). It is a research/prototyping repo, not a product. There is no build system, package manager, or test suite — work happens partly in Google Colab and partly as static files on the Mac.

The eventual pipeline is **four stages** (see `face_hashing_research_report.md` for the deep dive on tooling at each stage):

1. **Extract** — face → structured representation. Current choice: FLAME parameters via DECA (a dict of `shape`/`exp`/`pose`/`tex`/`detail`/`cam`/`light` tensors — the "facial features as JSON"). The report also evaluates ArcFace 512-d embeddings as an alternate/hybrid path.
2. **Transform** — the "hash function": `transform(features, key) -> features'`. **Not implemented yet.** Designed to be a hot-swappable strategy (e.g. `flame_shape_offset_v1`, `rotation_v1`). For FLAME, mutate `shape` (identity), preserve `exp`/`pose`.
3. **Reconstruct** — transformed params → rough face (FLAME mesh, or Arc2Face for embeddings).
4. **Photorealism + composite** — diffusion model (InstantID / PuLID / Arc2Face) to make it a photo and paste back into the original image.

> Note on "one-way": true cryptographic one-wayness is not achievable here — treat the transform as *obfuscation*, not encryption (see report Key Finding 3 and the RiDDLE paper for the closest reversible-with-key prior art).

## Current state — Stage 1 only

The active milestone — **photo → DECA → FLAME params + 3D mesh → browser viewer** — works end-to-end (verified 7/7 in Colab on 2026-05-29). A manual param-tweak slider (`pipeline.tweak`) previews the Stage-2 transform; the automated hash itself is not implemented yet.

`CONTEXT.md` is the authoritative "where am I right now" file — read it first, it reflects the actual working state and supersedes the older guide where they conflict.

## Two-machine architecture

There is no single runnable app. The system is split:

- **Machine A — Google Colab (the GPU).** Runs DECA. The repo has no `.ipynb`; the live notebook lives at the Colab URL inside `colab/DECA.ipynb - Colab.webloc`. macOS lacks an NVIDIA GPU and PyTorch3D won't build locally, which is *why* DECA runs in Colab.
- **Machine B — the Mac (the viewer).** A single static HTML page using Google's `<model-viewer>` web component to drag-rotate the `.glb`. No framework, no build step.

**Handoff** between the two runs through the Google Drive alias (`Drive Folder` → `My Drive/Face-Hashing`, which holds `Input/`, `Output/`, the FLAME model, and a `cache/` for big downloads): the Mac drops input photos there, Colab mounts Drive to read them and writes results back, and the Mac picks up the `.glb`/params from the synced folder. Drive also serves as the **cross-restart cache** so free-tier resets don't re-download the 434 MB DECA weights.

## Commands

**Run Stage 1 (in a Colab notebook).** DECA runs **in-kernel with no renderer**; `pipeline.py` wraps it so the notebook stays thin:

```python
!git clone https://github.com/tragicallyludicrous/face-hashing.git
import sys; sys.path.insert(0, "face-hashing")
import pipeline
pipeline.bootstrap()                  # clone DECA, install, patch, cache weights (idempotent)
deca, faces = pipeline.load_deca()    # renderer-free DECA + FLAME topology
pipeline.reconstruct(deca, faces)     # Drive Input/ photos -> Output/<name>/{.glb,_params.npz}
pipeline.tweak(deca, faces, "<name>") # mutate identity, re-export <name>_tweaked.glb
```

We do **not** use `demos/demo_reconstruct.py` or its CUDA rasterizer — `encode()` + the FLAME decoder produce the params + mesh without rendering. The cell-by-cell version (and the rationale) is in `face-hashing-setup.md`.

**Serve the viewer locally (on the Mac):**

```bash
cd viewer && python3 -m http.server 8080   # then open http://localhost:8080
```

`<model-viewer>` cannot load `.glb` over `file://`; it must be served over HTTP.

## Colab gotchas (re-run every session)

Colab is Python 3.12 (DECA targets 3.7–3.10) and resets on disconnect, so setup re-runs each session — `pipeline.bootstrap()` handles it idempotently. Because we run **in-kernel** (no `!python` subprocess), the chumpy fix is a simple in-process monkeypatch, applied *before* any `decalib` import:

```python
import inspect, numpy as np
if not hasattr(inspect, 'getargspec'): inspect.getargspec = inspect.getfullargspec
for a, r in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
    if not hasattr(np, a): setattr(np, a, r)
```

The detector also needs `LandmarksType._2D` → `TWO_D` patched in `decalib/datasets/detectors.py`. Weights are cached in Drive (`deca_model.tar` gdown'd **once** into `Face-Hashing/cache/` then copied locally each session; FLAME `generic_model.pkl` from `Face-Hashing/FLAME/...`). The `.glb` is built from FLAME verts + faces via `trimesh` with a flat gray vertex color (`[180,180,200,255]`) for the untextured "Skyrim" look. If you ever shell back out to `!python`, the in-kernel chumpy patch won't carry over — see the on-disk shim fallback in `CONTEXT.md`.

## Local-only / license-gated files (git-ignored)

`.gitignore` keeps these out of the repo — keep it that way:

- `inputs/` — **personal photos** of real people. These now live in the Google Drive alias (`Drive Folder` → `My Drive/Face-Hashing`) and sync to Colab from there; the local `inputs/` is empty.
- `outputs/` — DECA results (also handed off via the Drive alias; only `outputs/.gitkeep` is tracked).
- `colab/FLAME2020/` — FLAME 2020 model weights (~153 MB, registration-required, **non-commercial license**).
- `Drive Folder` — machine-specific macOS bookmark to the Drive folder.

Licensing: DECA's pretrained weights are research-only; FLAME 2020 is CC-BY with content restrictions.

## Doc map (authoritative order)

- `CONTEXT.md` — **current truth**: live state, the in-kernel/no-renderer decision, patches, design decisions. Start here.
- `face-hashing-setup.md` — the Stage-1 **procedural source of truth**: cell-by-cell notebook (in-kernel, no renderer), Drive caching for fast restarts, footguns. Rewritten 2026-05-29; current.
- `pipeline.py` — reusable Colab module (`bootstrap` / `load_deca` / `reconstruct` / `tweak`) that the thin notebook imports.
- `face_hashing_research_report.md` — deep reference on tools/approaches/pricing for all four stages. Consult when designing Stage 2+.
