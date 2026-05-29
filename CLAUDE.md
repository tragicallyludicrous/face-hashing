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

The active milestone is just: **photo → DECA → FLAME params + 3D mesh → browser viewer. No transform.** It proves the photo-to-structured-representation pipeline works.

`CONTEXT.md` is the authoritative "where am I right now" file — read it first, it reflects the actual working state and supersedes the older guide where they conflict.

## Two-machine architecture

There is no single runnable app. The system is split:

- **Machine A — Google Colab (the GPU).** Runs DECA. The repo has no `.ipynb`; the live notebook lives at the Colab URL inside `colab/DECA.ipynb - Colab.webloc`. macOS lacks an NVIDIA GPU and PyTorch3D won't build locally, which is *why* DECA runs in Colab.
- **Machine B — the Mac (the viewer).** A single static HTML page using Google's `<model-viewer>` web component to drag-rotate the `.glb`. No framework, no build step.

**Handoff** between the two runs through the Google Drive alias (`Drive Folder` → `My Drive/Face-Hashing`, which holds `inputs/` and `outputs/`): the Mac drops input photos there, Colab mounts Drive to read them and writes results back, and the Mac picks up the `.glb`/params from the synced folder.

## Commands

**Run DECA (in a Colab cell)** — note this diverges from the setup guide: PyTorch3D is skipped entirely and visualization is off:

```python
!python demos/demo_reconstruct.py -i TestSamples/examples -s outputs/examples \
    --saveObj True --saveMat True --saveVis False
```

`--saveMat True` produces the `.mat` of FLAME parameters (the "JSON"); `--saveObj True` produces the mesh; `--saveVis False` avoids the renderer.

**Serve the viewer locally (on the Mac):**

```bash
cd viewer && python3 -m http.server 8080   # then open http://localhost:8080
```

`<model-viewer>` cannot load `.glb` over `file://`; it must be served over HTTP.

## Colab gotchas (re-run every session)

Colab is Python 3.12 (DECA targets 3.7–3.10) and resets on disconnect, so its environment must be re-patched each session. Because each `!python` call is a fresh interpreter, **patches must be on disk (sed), not in-process monkey-patches.** The three required patches (from `CONTEXT.md`):

```bash
sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' /usr/local/lib/python3.12/dist-packages/chumpy/ch.py
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' /usr/local/lib/python3.12/dist-packages/chumpy/__init__.py
sed -i 's/LandmarksType\._2D/LandmarksType.TWO_D/g' /content/DECA/decalib/datasets/detectors.py
```

DECA needs three weight files: `deca_model.tar` (gdown), and FLAME 2020's `generic_model.pkl` (registration-gated, uploaded manually). `.obj` → `.glb` conversion is done with `trimesh`, applying a flat gray vertex color (`[180,180,200,255]`) for the untextured "Skyrim" look.

## Local-only / license-gated files (git-ignored)

`.gitignore` keeps these out of the repo — keep it that way (there are **no commits yet**, so a first commit is where this matters):

- `inputs/` — **personal photos** of real people. These now live in the Google Drive alias (`Drive Folder` → `My Drive/Face-Hashing`) and sync to Colab from there; the local `inputs/` is empty.
- `outputs/` — DECA results (also handed off via the Drive alias; only `outputs/.gitkeep` is tracked).
- `colab/FLAME2020/` — FLAME 2020 model weights (~153 MB, registration-required, **non-commercial license**).
- `Drive Folder` — machine-specific macOS bookmark to the Drive folder.

Licensing: DECA's pretrained weights are research-only; FLAME 2020 is CC-BY with content restrictions.

## Doc map (authoritative order)

- `CONTEXT.md` — **current truth**: live state, exact flags, applied patches, design decisions. Start here.
- `face-hashing-setup.md` — original Stage-1 step-by-step. Useful background but **partly stale** (it still tells you to install PyTorch3D and use `--rasterizer_type=pytorch3d` / `--saveVis True`; the actual workflow dropped all of that). Its own header flags it as behind.
- `face_hashing_research_report.md` — deep reference on tools/approaches/pricing for all four stages. Consult when designing Stage 2+.

## Known discrepancies to be aware of

- `CONTEXT.md`'s "repo layout" lists `colab/stage1_deca.ipynb` and `README.md`; neither exists yet (the notebook is the `.webloc` link; there is no README). It also still draws `index.html` under `viewer/models/` — it now lives at `viewer/index.html`.
