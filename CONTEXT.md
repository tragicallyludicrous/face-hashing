# Face Hashing — Project Context

## What I'm building

See below a lightly-modified version of the original deep-research prompt:

I want to build a personal-project program for exploratory/creative purposes. I call it Face Hashing, and here's the long and short of it:

In: a photograph (eventually video, but let's start with stills) of a person (eventually multiple people).
Out: the same photograph, but the person's face is different.

Here's the twist: the out face should always be the same face given the same input face. Under the hood, the transform should be, as any good hash:

Deterministic
One-way

Here's one way I picture it working, with a limited knowledge of the tools involved:

1. Some sort of facial recognition algorithm (perhaps the kind that can identify a specific face in, say, Apple Photos or the like) outputs the specifics of what makes that face unique. Pupillary distance, jaw shape, whatnot. In my brain this outputs as a JSON object or something.

2. Some sort of mathematical transform that changes that object's values (but not structure) such that they would represent the output of a different face. Ideally we could hotswap this 'hash function' as the project mutates.

3. Some way to turn this object into the rough draft version of the face. I picture something like the Skyrim character generator, but as granular as the data it receives.

4. A diffusion model to make this face photorealistic and composite it back into the original image.

Each step seems like a unique and interesting challenge, and I'd love to know what libraries, approaches, etc I should dig into in order to make this possible. Also open to other approaches (I imagine simply generating a seed from the facial-recognition algo, transforming that, and putting it into a comfyUI workflow might get up and running faster but it doesn't sound as interesting), though this one seems very interesting on many technical levels that I'd like to dig into.

## Stage 1 goal (current)

Photo → DECA → FLAME parameters (the "JSON") → 3D mesh → browser viewer.
No transform yet. Just proving the pipeline.

## Architecture (eventual, 4 stages)

1. Face → structured representation (FLAME params via DECA)
2. Deterministic transform on those params (the "hash function", hot-swappable)
3. Reconstruct a rough face from transformed params
4. Diffusion model for photorealism + composite back into original image

## Environment

- macOS Apple Silicon, no NVIDIA GPU
- Running DECA in Google Colab (T4 free tier) to avoid PyTorch3D install pain on Mac
- Viewer is local: HTML + <model-viewer> served via `python3 -m http.server`

## Repo layout

```
face-hashing/
├── CLAUDE.md                       # guidance for Claude Code
├── CONTEXT.md                      # this file — current truth
├── face-hashing-setup.md           # Stage-1 procedural guide (cells + Drive caching)
├── face_hashing_research_report.md # deep tool reference for Stages 2–4
├── pipeline.py                     # reusable Colab module (bootstrap/load_deca/reconstruct/tweak)
├── colab/
│   └── DECA.ipynb - Colab.webloc   # link to the live Colab notebook (no .ipynb in repo yet)
├── outputs/.gitkeep                # results handed off via Drive; not committed
├── viewer/
│   ├── index.html                  # the <model-viewer> page
│   └── models/.gitkeep             # .glb files go here (git-ignored)
└── .gitignore
```
(Personal photos, FLAME weights, and the `Drive Folder` alias are git-ignored.)

## Where I am right now (2026-05-29)

**Stage 1 runs in-kernel with NO renderer** (see Key decision below). The flow:

- In the Colab notebook process, construct DECA with its rasterizer neutered
  (`DECA._setup_renderer = lambda self, m: None`), then call `deca.encode(img)` and the FLAME
  decoder `deca.flame(shape, exp, pose)` directly → vertices → trimesh `.glb` + params `.npz`.
  This replaces `demos/demo_reconstruct.py` entirely (no rasterizer, no `--saveVis`).
- chumpy fixed with an **in-kernel monkeypatch** (inspect.getargspec + numpy aliases) —
  sufficient now that we don't shell out to a `!python` subprocess.
- face-alignment `LandmarksType._2D` → `TWO_D` patched in DECA's `detectors.py`.
- Weights cached in Drive: `deca_model.tar` gdown'd once into `Face-Hashing/cache/` and copied
  locally each session; FLAME `generic_model.pkl` copied from `Face-Hashing/FLAME/FLAME2020/`.
- Exact cells: `face-hashing-setup.md` §3. Reusable module: `pipeline.py`.
- Verified in Colab: the in-kernel path produces `.glb` + params (the `flame.faces_tensor`
  attribute and `flame()` kwargs are confirmed on this DECA build; first image succeeded).
- Inputs are downscaled to ≤1024 px (+ EXIF-rotated) before the FAN detector — it otherwise
  runs on the full image and OOMs a free-tier T4 on large phone photos — and GPU memory is freed
  per image. `.jpeg` is handled via an explicit file list (TestData's glob misses it).

## Key decision (2026-05-29): in-kernel, no renderer, keep DECA

After fighting DECA's build on modern Colab (chumpy, face-alignment, and a custom CUDA
rasterizer that hardcodes gcc-7), three research agents converged on:

- **The rasterizer is optional.** Our deliverables (param dict + mesh) come from `encode()`
  (a plain ResNet) and the FLAME decoder — neither renders. The CUDA build only fires from
  `set_rasterizer('standard')` in `DECA.__init__`; neuter it and the build is gone. This is
  also why dropping the renderer doesn't hurt the Skyrim-slider demo — that demo is just
  params → FLAME decoder → mesh → `.glb`, never a render.
- **Don't switch tools to escape the build.** Every maintained successor (SMIRK CVPR'24,
  EMOCA→INFERNO, MICA) depends on the SAME PyTorch3D and targets equally-old envs. FLAME's
  shape/exp/pose is the cleanest match to "mutate identity, preserve expression", so DECA/FLAME
  stays. SMIRK is bookmarked as a later *quality* upgrade (MIT, better expressions), not an
  infra fix. MediaPipe is the escape hatch only if install-simplicity ever beats having a real
  identity subspace — it has none, so it would weaken Stage 2.
- **Run in-kernel.** Avoids the subprocess-loses-monkeypatch problem and is the setup Stage 2
  needs anyway (load params → transform → re-decode, all live in the kernel).

## Patches needed each cold start (ephemeral runtime; re-run after every reset)

`site-packages` and `/content` are wiped on reset (Drive persists). `pipeline.bootstrap()`
re-applies these idempotently. With the in-kernel approach they're simple:

**1. chumpy — in-kernel monkeypatch, run BEFORE importing decalib/chumpy:**

```python
import inspect, numpy as np
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for a, r in [('bool',bool),('int',int),('float',float),('complex',complex),
             ('object',object),('str',str),('unicode',str)]:
    if not hasattr(np, a): setattr(np, a, r)
```

**2. face-alignment** — replace `LandmarksType._2D` → `LandmarksType.TWO_D` in
`/content/DECA/decalib/datasets/detectors.py` (a hard reference resolved at import).

> **Fallback, only if you ever run DECA as a `!python` subprocess again:** the in-kernel
> patch won't carry into a fresh interpreter (that's what bit us with `demo_reconstruct.py` —
> `AttributeError: module 'inspect' has no attribute 'getargspec'` at `chumpy/ch.py:1203`).
> Then prepend an on-disk shim to chumpy's `__init__.py` so it runs on every `import chumpy`:
>
> ```python
> import os, chumpy
> shim = (
>     "import numpy as _np, inspect as _inspect\n"
>     "if not hasattr(_inspect, 'getargspec'):\n"
>     "    _inspect.getargspec = _inspect.getfullargspec\n"
>     "for _a, _r in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:\n"
>     "    if not hasattr(_np, _a):\n"
>     "        setattr(_np, _a, _r)\n"
>     "# --- end DECA-on-py312 shim ---\n"
> )
> p = os.path.join(os.path.dirname(chumpy.__file__), '__init__.py')
> s = open(p).read()
> if 'DECA-on-py312 shim' not in s: open(p, 'w').write(shim + s)
> ```
>
> It *restores* the numpy aliases rather than stripping the `from numpy import bool, …` line,
> which chumpy uses internally during the FLAME `pickle.load`. Not needed for the in-kernel path.

## Known design decisions

- Keep DECA/FLAME over switching (successors share the PyTorch3D dependency; FLAME params fit
  the hash design) — see Key decision above.
- Run DECA in-kernel and skip the rasterizer (params + mesh need no rendering).
- Untextured coarse mesh (gray vertex color) for the Skyrim aesthetic.
- `<model-viewer>` for the browser viewer (one HTML tag, no three.js code).
- FLAME verts + faces → `.glb` via trimesh (model-viewer doesn't load `.obj`).
- Google Drive doubles as the cross-restart cache: big/slow downloads (the 434 MB weights) live
  there; only cheap steps (repo clone, pip, patches) are redone each session. Do NOT persist
  `site-packages` to Drive — native libs load flakily over the FUSE mount.

## Open questions / known footguns

- Stage 2: implement `transform(params, key) -> params'` as a hot-swappable strategy
  (start: seeded Gaussian offset on `shape`, clamp ±2σ; preserve `exp`/`pose`). Extend
  `pipeline.default_mutation` / add a registry.
- DECA's `TestData` directory glob misses `.jpeg` (only `*.jpg/*.png/*.bmp`) → silent 0-image
  no-op; pass an explicit file list. The FAN detector runs on the full image and OOMs a T4 on
  large phone photos → downscale to ≤1024 px before detection and free GPU memory per image
  (after an OOM, restart the runtime — IPython's saved traceback pins the GPU memory).
- Colab is Python 3.12 (DECA targets 3.7–3.10). You can NO LONGER pin Colab to 3.10 (aged out
  of the 1-year runtime window); 3.11 is the oldest selectable and the choice doesn't persist
  across sessions — so keep setup idempotent instead.
- Correction to an earlier note: a prebuilt PyTorch3D wheel DOES now exist for modern stacks
  (MiroPsota builder, through Torch 2.6 / cu126 / py3.14) — relevant only if you ever want
  rendering back; the in-kernel path needs no PyTorch3D at all.

## Reference

- `face-hashing-setup.md` — the current Stage-1 procedural source of truth (cells + caching).
- `pipeline.py` — reusable module (`bootstrap` / `load_deca` / `reconstruct` / `tweak`).
- `face_hashing_research_report.md` — deep tool reference for Stages 2–4.
