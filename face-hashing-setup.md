# Face Hashing — Project Setup Guide (Stage 1)

**Goal of this milestone:** photo → DECA → FLAME parameters (your "JSON object" of facial features) → 3D mesh → drag-rotate it in the browser, then tweak the parameters and watch the face change. **No hash transform yet** — Stage 1 just proves we can get from a photo to a structured 3D representation we can play with.

> **Status (2026-05-29): Stage 1 works end-to-end** — every input photo reconstructs to `.glb` + params on a free-tier T4. **The way to run it is the `pipeline.py` notebook (§3)** — a handful of `pipeline.*` calls. The manual cell-by-cell version lives in the appendix (§7) as the reference for what those calls do under the hood.
>
> Rewritten 2026-05-29 around the *in-kernel, no-renderer* approach. The old flow's PyTorch3D / `--rasterizer_type` / `--saveVis` / `demo_reconstruct.py` instructions are abandoned (see §2). If you see them in an old notebook, they're stale.

This doc is the **procedural source of truth**. `CONTEXT.md` is the higher-level "where am I" doc; `pipeline.py` is the actual code; `face_hashing_research_report.md` is the deep tool reference for later stages.

---

## How this guide is organized

1. The mental model — what the pieces are and why
2. Why in-kernel + no renderer — the key decision
3. **Running it — the `pipeline.py` notebook** (start here to actually run)
4. The viewer (on the Mac)
5. Known footguns
6. What to do when it works (incl. the param-tweak / "Skyrim-slider" demo)
7. Appendix — what `pipeline.py` runs, cell by cell
8. Keeping the notebook and these docs in sync

---

## 1. The mental model

Two-machine system:

- **Machine A — Google Colab (the GPU).** Runs DECA. You point it at photos in Google Drive; it returns FLAME parameters + a 3D mesh (`.glb`) back to Drive.
- **Machine B — the Mac (the viewer).** A single static HTML page (`viewer/index.html`) using Google's `<model-viewer>` to drag-rotate the `.glb`. No build step.

**Handoff is via the Google Drive alias** (`Drive Folder` → `My Drive/Face-Hashing`, holding `Input/`, `Output/`, the FLAME model, and a `cache/` for big downloads). The Mac drops photos in `Input/`; Colab mounts Drive, reads them, writes `.glb` + params to `Output/`; the synced folder brings them back to the Mac. Drive also doubles as the **cross-restart cache** (§3) so free-tier resets don't re-download large files.

### Glossary

- **FLAME** — a parametric 3D face model. Feed it ~100 **shape** coefficients (identity) + ~50 **expression** coefficients + 6 **pose** values (jaw + global rotation) and it outputs a fixed-topology **5023-vertex mesh**. The "rig" every face is a setting of.
- **DECA** — a network that regresses FLAME parameters from a single photo. Two halves matter to us:
  - **`encode(image)`** → the parameter dict (`shape`, `exp`, `pose`, `tex`, `cam`, `light`, `detail`). This is a plain ResNet. **No renderer.**
  - **the FLAME decoder** (`deca.flame(shape, exp, pose)`) → mesh **vertices**. **No renderer.**
- **DECA's rasterizer / renderer** — a *separate* component that draws flat 2D preview images of the mesh (shaded, lit). It's the thing that needs a custom CUDA build. **We don't use it** — see §2.
- **`.glb`** — binary glTF, the format `<model-viewer>` wants. We build it from FLAME vertices + faces with `trimesh`.
- **`<model-viewer>`** — Google web component; one HTML tag gives a drag-rotatable 3D viewer.

The crucial distinction: **the interactive 3D face you drag in the browser is the mesh (vertices + faces), not a render.** DECA's rasterizer only makes throwaway 2D preview PNGs. So dropping it costs us nothing for Stage 1 *or* the slider demo.

---

## 2. Why in-kernel + no renderer (read this once)

DECA is a 2021 repo targeting Python 3.7–3.10. On current Colab (Python 3.12, Torch 2.x, CUDA 12) the old "install PyTorch3D and compile the rasterizer" path is a tarpit: no matching prebuilt PyTorch3D wheel for the default runtime, and DECA's hand-rolled CUDA rasterizer hardcodes `gcc-7` (absent on Colab). We burned hours there.

The escape: **we never needed the rasterizer.** Our Stage-1 deliverables are the parameter dict (`.npz`) and the mesh (`.glb`), both produced by `encode()` + the FLAME decoder, neither of which touches the rasterizer. The CUDA build only fires from one call (`set_rasterizer('standard')`) inside `DECA.__init__`. We neuter that call and call the FLAME decoder directly.

Running **in-kernel** (in the notebook process, not via `!python …`) buys two more things:

1. **The chumpy fix becomes a simple in-kernel monkeypatch** — no on-disk shim needed, because there's no fresh `!python` subprocess to lose the patch. (A subprocess starts a clean interpreter that wouldn't see in-memory patches; that was the whole reason the old flow needed on-disk `sed`s.)
2. **It's exactly the setup Stage 2 needs** — to hash and re-render you'll want the FLAME decoder live in the kernel anyway. The tweak demo (§6) is then a few lines.

What we give up: DECA's baked 2D preview images. Those are strictly worse than the live `<model-viewer>`.

---

## 3. Running it — the `pipeline.py` notebook

All the logic lives in `pipeline.py` in this repo. The Colab notebook is just a thin runner — four cells:

```python
# Cell 1 — get the repo + import (the `pull` branch picks up later edits)
import os, sys, importlib
REPO = "/content/face-hashing"
if not os.path.isdir(REPO):
    !git clone https://github.com/tragicallyludicrous/face-hashing.git "{REPO}"
else:
    !git -C "{REPO}" pull
sys.path.insert(0, REPO)
import pipeline; importlib.reload(pipeline)
```

```python
# Cell 2 — cold-start setup (idempotent: safe to re-run every session)
pipeline.bootstrap()
```

```python
# Cell 3 — build a renderer-free DECA, reconstruct every photo in Drive Input/
deca, faces = pipeline.load_deca()
names = pipeline.reconstruct(deca, faces)   # -> Output/<name>/{<name>.glb, <name>_params.npz}
names
```

```python
# Cell 4 — tweak / compare (optional)
pipeline.tweak(deca, faces, names[0])                      # mutate identity -> <name>_tweaked.glb
for n in names: pipeline.export_neutral(deca, faces, n)    # frontal+neutral -> <name>_neutral.glb (aligned compare)
```

What each call does:

| Call | Does |
|---|---|
| `bootstrap()` | mount Drive · clone DECA if missing · `pip install` deps · patch detector + chumpy · fetch the 434 MB weights (gdown **once** into Drive `cache/`, copied locally each session). Idempotent — re-run on every cold start. |
| `load_deca()` | build DECA with the CUDA rasterizer neutered; returns `(deca, faces)`. |
| `reconstruct(deca, faces)` | every photo in Drive `Input/` → `encode()` → FLAME decoder → `.glb` + `_params.npz` in `Output/`. Downscales inputs to ≤1024 px (detector-OOM guard) and handles `.jpeg`. Returns the list of names. |
| `tweak(deca, faces, name)` | re-decode with `shape` mutated (identity), `exp`/`pose` kept → `<name>_tweaked.glb`. The hook for Stage 2 (`pipeline.default_mutation`). |
| `export_neutral(deca, faces, name)` | re-decode with pose (and expression) zeroed → `<name>_neutral.glb`, so faces line up for shape comparison (§4). |

**Fast restarts.** Free Colab resets often and wipes the runtime disk (Drive persists). The rule baked into `bootstrap()`: **cache big/slow downloads in Drive, redo only the cheap stuff.** The weights + FLAME model live in Drive and are copied in on boot (fast internal copy, *not* an internet re-download); the repo clone + pip install are cheap and just re-run. **Don't** persist `site-packages` to Drive — native libs (opencv/kornia/face-alignment) load flakily over the FUSE mount and break when Colab bumps Python.

**Drive layout** (`My Drive/Face-Hashing/`):
```
Input/                            # your photos
Output/                           # results (.glb, _params.npz)
FLAME/FLAME2020/generic_model.pkl # registration-gated; placed here once
cache/deca_model.tar              # gdown'd once, reused every session
```

**To change the pipeline:** edit `pipeline.py` in the repo, push, then in Cell 1 the `pull` + `importlib.reload(pipeline)` picks it up. If you need to debug a single step in isolation, §7 has the raw cells each function runs.

---

## 4. The viewer (on the Mac)

The page lives at `viewer/index.html`; `.glb` files go in `viewer/models/`. The viewer **auto-discovers** every `.glb` in `models/` — it fetches the directory listing `http.server` serves and populates the dropdown — so you just copy files in and reload, no HTML editing.

1. Copy `.glb`s out of the synced Drive `Output/<name>/` into `viewer/models/` (e.g. `<name>.glb` and `<name>_tweaked.glb` for an A/B).
2. Serve over HTTP — `<model-viewer>` won't load `.glb` over `file://`:
   ```bash
   cd viewer && python3 -m http.server 8080   # open http://localhost:8080
   ```
3. Pick a model from the dropdown and drag to rotate. Entries are sorted, so `<name>` and `<name>_tweaked` sit next to each other for easy comparison.

Filenames with spaces (e.g. the `hoga …` one) work as-is — the directory listing URL-encodes them. If the dropdown stays empty, you opened the page over `file://` (the listing fetch only works over `http://`) — use the `http.server` command above.

**Comparing faces (making them line up).** DECA bakes each photo's head pose into its mesh, so faces from different photos point different directions and won't align no matter how you rotate the camera. For an apples-to-apples *shape* comparison, export neutral (frontal, expression-zeroed) meshes and view those instead:

```python
for n in names: pipeline.export_neutral(deca, faces, n)   # writes <name>_neutral.glb each
```

Copy the `_neutral.glb`s into `viewer/models/` — they all share one canonical orientation, so flipping between them shows pure identity differences. The viewer also keeps your camera angle fixed across model switches (and auto-rotate is off), so the viewpoint stays put while you compare.

---

## 5. Known footguns

- **Colab resets wipe everything installed.** Free-tier kills idle sessions (~90 min) and the runtime's filesystem is ephemeral. Re-run `pipeline.bootstrap()` on a fresh session (~a few minutes). Your photos/outputs survive because they live in Drive. **You can no longer pin Colab to Python 3.10** (it aged out of Colab's 1-year runtime window); 3.11 is the oldest selectable and the choice doesn't persist, so just rely on `bootstrap()` being idempotent.
- **chumpy import errors** (`module 'inspect' has no attribute 'getargspec'`, or numpy `bool`/`str` AttributeErrors) mean the chumpy patch didn't run before DECA was built. `bootstrap()` orders this for you; you only hit it if you run the raw appendix cells out of order (run the chumpy cell before `load_deca`).
- **`reconstruct` silently does nothing** (no `ok …` prints, no error) → it found 0 images. DECA's own directory glob only matches `*.jpg/*.png/*.bmp`, so `.jpeg`/uppercase extensions would yield an empty set — `pipeline.reconstruct` avoids this by building an explicit, case-insensitive file list. If you still see 0, check the photos are directly inside Drive `Input/`.
- **CUDA out of memory in the FAN detector** (OOM at face detection, trying to allocate many GiB). The detector runs on the *full-resolution* image, and phone photos blow the T4's ~14.5 GB. `pipeline.reconstruct` downscales each image to ≤1024 px first and frees GPU memory per image. If an OOM still leaves the GPU pinned (IPython's saved traceback holds the tensors), `Runtime → Restart`, then re-run `bootstrap()` → `load_deca()` → `reconstruct()`.
- **Face detection quality.** DECA wants a clear, frontal-ish face; profile shots, sunglasses, hands/hair over the face, and tiny faces degrade it. EXIF rotation on iPhone photos is a common silent failure — `reconstruct` applies `exif_transpose`, but if a detect still fails, re-export the photo. One face per image (DECA reconstructs one).
- **"It doesn't really look like me."** Expected, not a bug — single-image FLAME regression is constrained to FLAME's learned face manifold (generic noses, puffy chins). Recognizable but stylized. MICA gives more metric identity if this ever matters; don't switch yet.
- **`deca.flame(...)` kwargs / `faces_tensor`** (verified on the current DECA build; portability note for other revisions). If `faces_tensor` is missing, `load_deca` falls back to `deca_cfg.model.topology_path` (the bundled `head_template.obj`). If the `flame()` kwargs differ, check `decalib/models/FLAME.py`'s `forward` signature.

---

## 6. What to do when it works

### Sanity-check identity *consistency* (`pipeline.consistency_report`)
For a hash, the same person must land on the same identity vector across photos. Label which photos are which person (≥2 per person for at least one), then measure:

```python
!pip install -q insightface onnxruntime    # ArcFace baseline; CPU is fine
import importlib, pipeline; importlib.reload(pipeline)
pipeline.consistency_report({
    '000002540007': 'p1', 'IMG_1237': 'p1',   # same person, multiple photos
    'IMG_3642': 'p2', 'IMG_4863': 'p2',
})
```

It prints, per extractor, mean intra-/inter-person distance, the **inter/intra separation ratio** (higher = better), and a verification **AUC** (1.0 = perfect). It compares **DECA shape** (the current carrier, L2) against the **ArcFace `buffalo_l`** embedding (cosine) — the consistency *ceiling*. Expect ArcFace to separate identities far more cleanly: DECA shape drifts because it optimizes per-photo reconstruction, not identity invariance. This is the read that says whether the identity carrier is stable enough to hash — and the case for making ArcFace (bridged to FLAME via MICA) the identity backbone in a later stage.

### The tweak intuition (`pipeline.tweak`)
`tweak` flips/scales `shape` coefficients and re-decodes. The first few coefficients are the loud perceptual modes (face width, head length, nose prominence-ish); later ones are finer. Knowing which dims are loud vs. quiet is exactly what tells you what a Stage-2 hash should mutate hard vs. leave alone. Pass your own `mutate=` to `tweak` to experiment.

### Next milestones
1. ✅ Photo → params + mesh → viewer.
2. ✅ Manually tweak params and re-render (`pipeline.tweak`).
3. **Stage 2 — the hash.** Replace the hand-edits with `transform(params, key) -> params'` behind a hot-swappable strategy interface (start: seeded Gaussian offset on `shape`, clamped to ±2σ; preserve `exp`/`pose`). Extend `pipeline.default_mutation` / add a registry. See the research report.
4. **Stage 4 — photorealism.** Layer Arc2Face / InstantID / PuLID once the 3D pipeline feels good.

---

## 7. Appendix — what `pipeline.py` runs, cell by cell

You normally **don't type these** — `bootstrap` / `load_deca` / `reconstruct` / `tweak` do all of it (§3). They're here as (a) the reference for what each function does, and (b) raw cells to drop into a notebook when you need to debug one step in isolation. Mapping: Cells 1–4 ≈ `bootstrap()`, Cell 5 ≈ `load_deca()` + `reconstruct()`, Cell 6 ≈ `tweak()`. Order matters — chumpy (Cell 4) before DECA is constructed (Cell 5).

### Cell 1 — mount Drive + define paths

```python
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/Face-Hashing'
CACHE = f'{DRIVE}/cache'; os.makedirs(CACHE, exist_ok=True)
IN, OUT = f'{DRIVE}/Input', f'{DRIVE}/Output'
```

### Cell 2 — clone DECA, install deps, patch the detector

```python
import os
if not os.path.isdir('/content/DECA'):
    !git clone https://github.com/yfeng95/DECA.git
%cd /content/DECA
# DECA's pins are ancient; install what Colab lacks, leave Torch/CUDA alone.
!pip install -q chumpy yacs==0.1.8 face-alignment ninja kornia==0.6.12 scikit-image opencv-python PyYAML trimesh gdown
# face-alignment renamed LandmarksType._2D -> TWO_D; DECA's detector still uses the old name.
!sed -i 's/LandmarksType\._2D/LandmarksType.TWO_D/g' decalib/datasets/detectors.py
```

> **Optional, to shave the pip minute:** cache wheels in Drive once with `!pip download -q -d "{CACHE}/wheels" <same list>`, then install with `!pip install -q --no-index --find-links="{CACHE}/wheels" <same list>`. Wheels are Python-version-specific — delete `cache/wheels` if Colab bumps Python; fall back to plain `pip install` if `--no-index` errors.

### Cell 3 — weights, cached in Drive (gdown runs ONCE, ever)

```python
import os
os.makedirs('/content/DECA/data', exist_ok=True)
deca_tar = f'{CACHE}/deca_model.tar'
if not os.path.exists(deca_tar):                 # first session ever; skipped on every restart after
    !gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O "{deca_tar}"
!cp -n "{deca_tar}" /content/DECA/data/deca_model.tar          # Drive -> local; -n = don't re-copy
!cp -n "{DRIVE}/FLAME/FLAME2020/generic_model.pkl" /content/DECA/data/generic_model.pkl
!ls -lh data/deca_model.tar data/generic_model.pkl
```

### Cell 4 — chumpy patch (in-kernel; before DECA is constructed)

```python
import inspect, numpy as np
# chumpy is unmaintained: uses inspect.getargspec (removed in 3.11+) and numpy aliases (removed in numpy 2.x).
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for _a, _r in [('bool', bool), ('int', int), ('float', float),
               ('complex', complex), ('object', object), ('str', str), ('unicode', str)]:
    if not hasattr(np, _a):
        setattr(np, _a, _r)
import chumpy  # must succeed here, before DECA loads the FLAME .pkl (which unpickles chumpy objects)
print("chumpy", chumpy.__version__)
```

### Cell 5 — build a renderer-free DECA and reconstruct

```python
import torch, trimesh, numpy as np, os, gc
from PIL import Image, ImageOps
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

DECA._setup_renderer = lambda self, model_cfg: None   # no set_rasterizer() -> no CUDA build / no PyTorch3D
deca_cfg.model.use_tex = False                         # untextured coarse mesh is all Stage 1 needs
device = 'cuda'
deca = DECA(config=deca_cfg, device=device)            # ResNet encoder + FLAME; NO renderer

faces = deca.flame.faces_tensor.cpu().numpy()          # FLAME topology, from FLAME not the renderer
# fallback: faces = trimesh.load(deca_cfg.model.topology_path, process=False).faces

IN  = '/content/drive/MyDrive/Face-Hashing/Input'
OUT = '/content/drive/MyDrive/Face-Hashing/Output'

# Downscale (+ EXIF-rotate) before detection: the FAN detector runs on the FULL image and OOMs a
# T4 on big phone photos; DECA only needs ~224 px. Explicit list also sidesteps the .jpeg-missing glob.
exts, MAXSIDE, TMP = ('.jpg', '.jpeg', '.png', '.bmp'), 1024, '/content/_resized'
os.makedirs(TMP, exist_ok=True)
paths = []
for f in sorted(os.listdir(IN)):
    if os.path.splitext(f)[1].lower() not in exts: continue
    im = ImageOps.exif_transpose(Image.open(os.path.join(IN, f)).convert('RGB'))
    im.thumbnail((MAXSIDE, MAXSIDE))
    p = os.path.join(TMP, os.path.splitext(f)[0] + '.jpg'); im.save(p, quality=95); paths.append(p)

testdata = datasets.TestData(paths, iscrop=True, face_detector='fan')
for i in range(len(testdata)):
    with torch.no_grad():
        d = testdata[i]; name = d['imagename']
        images = d['image'].to(device)[None, ...]
        codedict = deca.encode(images)                                  # photo -> params
        verts, _, _ = deca.flame(shape_params=codedict['shape'],        # params -> mesh verts
                                 expression_params=codedict['exp'],
                                 pose_params=codedict['pose'])
    os.makedirs(f'{OUT}/{name}', exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[180, 180, 200, 255])
    mesh.export(f'{OUT}/{name}/{name}.glb')
    params = {k: v.detach().cpu().numpy() for k, v in codedict.items() if torch.is_tensor(v) and k != 'images'}
    np.savez(f'{OUT}/{name}/{name}_params.npz', **params)
    del images, codedict, verts; gc.collect(); torch.cuda.empty_cache()
    print('ok', name, '| shape', params['shape'].shape)
```

If you hit `AttributeError: 'DECA' object has no attribute 'render'`, something is calling the renderer — we call `deca.flame(...)` directly to avoid that, so don't call `deca.decode(...)`.

### Cell 6 — the param-tweak ("Skyrim-slider") demo

```python
import numpy as np, torch, trimesh
name = '<one of your imagenames>'
p = np.load(f'{OUT}/{name}/{name}_params.npz')
shape = torch.tensor(p['shape']).to(device).clone()
exp   = torch.tensor(p['exp']).to(device)
pose  = torch.tensor(p['pose']).to(device)

shape[0, 0] *= -1.0   # flip the biggest shape PC (~face width); keep exp/pose so smile/angle persist
shape[0, 1] *= 1.5
with torch.no_grad():
    verts, _, _ = deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
mesh = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), faces=faces, process=False)
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[200, 180, 180, 255])
mesh.export(f'{OUT}/{name}/{name}_tweaked.glb')
print('exported tweaked glb')
```

---

## 8. Keeping the notebook and these docs in sync

The logic lives in `pipeline.py` in the repo, **not** in the notebook — so the notebook stays a thin 4-cell runner that rarely changes. That's the sync mechanism: Claude edits `pipeline.py` directly (clean diffs, no notebook-JSON noise), you push, and Cell 1's `git -C face-hashing pull` + `importlib.reload(pipeline)` pulls it into Colab. Data still flows through Drive.

Two optional niceties:
- **Version the thin notebook:** in Colab, *File → Save a copy in GitHub* commits the `.ipynb` here, so there's a canonical copy (and Claude can read the real cells). Notebooks are code, not gitignored data — safe to commit; photos/weights stay in Drive.
- The live notebook URL is bookmarked at `colab/DECA.ipynb - Colab.webloc`.

When you change `pipeline.py`: push, then re-run Cell 1 (`pull` + `reload`) in Colab.
