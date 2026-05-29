# Face Hashing — Project Setup Guide (Stage 1)

**Goal of this milestone:** photo → DECA → FLAME parameters (your "JSON object" of facial features) → 3D mesh → drag-rotate it in the browser. Then tweak the parameters and watch the face change (the "Skyrim slider" moment). **No hash transform yet** — Stage 1 just proves we can get from a photo to a structured 3D representation we can play with.

> **This guide was rewritten 2026-05-29** around the *in-kernel, no-renderer* approach. The earlier version told you to install PyTorch3D and JIT-compile DECA's CUDA rasterizer; that path is abandoned (see "Why in-kernel" below). If you find any PyTorch3D / `--rasterizer_type` / `--saveVis` instructions in your notebook, they're from the old flow.
>
> **Status (2026-05-29): Stage 1 works end-to-end** — every input photo reconstructs to `.glb` + params on a free-tier T4. The day-to-day way to run it is the **3-cell `pipeline.py` notebook** (§7); the cell-by-cell walkthrough in §3 is the under-the-hood reference for what `pipeline.py` does.

This doc is the **procedural source of truth** for building/modifying the Colab notebook. `CONTEXT.md` is the higher-level "where am I right now" doc; `face_hashing_research_report.md` is the deep tool reference for later stages.

---

## How this guide is organized

1. The mental model — what the pieces are and why
2. **Why in-kernel + no renderer** — the key decision this rewrite is built on
3. The notebook, cell by cell
4. The viewer (on the Mac)
5. Known footguns
6. What to do when it works (incl. the Skyrim-slider demo)
7. Keeping the notebook, these docs, and Claude in sync

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
2. **It's exactly the setup Stage 2 needs** — to hash and re-render you'll want the FLAME decoder live in the kernel anyway. The slider demo (§6) is then a few lines.

What we give up: DECA's baked 2D preview images. Those are strictly worse than the live `<model-viewer>`.

---

## 3. The notebook, cell by cell

**Built for fast restarts.** Free Colab resets often and wipes the runtime's disk each time, so the rule is: **cache anything big and slow in Drive; only re-do what's cheap.** The 434 MB DECA weights and the (registration-gated) FLAME model live in Drive permanently and are copied into the runtime on boot — a fast Google-internal copy, not an internet re-download. The DECA repo (~23 MB) and the pip packages are cheap, so we just re-fetch those. Every setup cell is **idempotent** (safe to re-run, does the minimum).

> **Don't persist the installed packages (`site-packages`) to Drive.** Classic Colab footgun: native/compiled libs (opencv, kornia, face-alignment) load slowly and flakily over the Drive FUSE mount and break the moment Colab bumps Python. Re-`pip install` each session (~1–2 min); cache only the big *downloads*.

Drive layout this assumes, under `My Drive/Face-Hashing/`:
```
Input/                            # your photos
Output/                           # results (.glb, _params.npz)
FLAME/FLAME2020/generic_model.pkl # registration-gated; you placed it here once
cache/deca_model.tar              # gdown'd once, reused every session
```

Order matters: mount Drive first (Cell 1); run the chumpy patch (Cell 4) before DECA is constructed (Cell 5).

> **Portability note:** two DECA API details (the FLAME faces attribute, the `flame()` kwargs) vary by revision; both are verified on the current build, with fallbacks noted inline.

### Cell 1 — mount Drive + define paths

```python
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/Face-Hashing'
CACHE = f'{DRIVE}/cache'; os.makedirs(CACHE, exist_ok=True)
IN, OUT = f'{DRIVE}/Input', f'{DRIVE}/Output'
```

### Cell 2 — clone DECA, install deps, patch the detector (cheap; re-run each session)

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

> **Optional, to shave the pip minute:** cache the wheels in Drive once —
> `!pip download -q -d "{CACHE}/wheels" chumpy yacs==0.1.8 face-alignment ninja kornia==0.6.12 scikit-image opencv-python PyYAML trimesh gdown` —
> then install with `!pip install -q --no-index --find-links="{CACHE}/wheels" <same list>`. Wheels are Python-version-specific: if Colab bumps Python, delete `cache/wheels` and re-download; if `--no-index` errors on a missing dep, fall back to plain `pip install`.

### Cell 3 — weights, cached in Drive (gdown runs ONCE, ever)

```python
import os
os.makedirs('/content/DECA/data', exist_ok=True)
deca_tar = f'{CACHE}/deca_model.tar'
if not os.path.exists(deca_tar):                 # first session ever; skipped on every restart after
    !gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O "{deca_tar}"
# Drive -> local copy each session (fast, internal; loading from local disk is reliable). -n = don't re-copy.
!cp -n "{deca_tar}" /content/DECA/data/deca_model.tar
!cp -n "{DRIVE}/FLAME/FLAME2020/generic_model.pkl" /content/DECA/data/generic_model.pkl
!ls -lh data/deca_model.tar data/generic_model.pkl
```

> To skip even the local copy, set `deca_cfg.pretrained_modelpath = deca_tar` in Cell 5 and load straight from Drive. That moves fewer bytes, but `torch.load` then reads 434 MB over the FUSE mount, which can be slower/flakier than a local read — the copy is the safer default.

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

> If you ever go back to running DECA as a `!python` subprocess, this in-kernel patch will NOT carry over (a fresh interpreter won't see it) — you'd need the on-disk shim in `CONTEXT.md`. Staying in-kernel keeps it simple.

### Cell 5 — build a renderer-free DECA and reconstruct

This is the cell that replaces the old `!python demos/demo_reconstruct.py …`.

```python
import torch, trimesh, numpy as np, os
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

# 1) Neuter the renderer so DECA() never calls set_rasterizer() -> no CUDA build, no PyTorch3D.
DECA._setup_renderer = lambda self, model_cfg: None

# 2) Untextured coarse mesh is all we need for Stage 1.
deca_cfg.model.use_tex = False

device = 'cuda'
deca = DECA(config=deca_cfg, device=device)   # builds the ResNet encoder + FLAME; NO renderer

# 3) FLAME topology (faces) — needed to turn vertices into a mesh; comes from FLAME, not the renderer.
faces = deca.flame.faces_tensor.cpu().numpy()
# (fallback if faces_tensor is absent on your DECA build:)
# faces = trimesh.load(deca_cfg.model.topology_path, process=False).faces

IN  = '/content/drive/MyDrive/Face-Hashing/Input'
OUT = '/content/drive/MyDrive/Face-Hashing/Output'

# Downscale (+ EXIF-rotate) into a temp dir BEFORE detection: the FAN detector runs on the FULL
# image and OOMs a T4 on big phone photos; DECA only needs ~224 px around the face. The explicit
# list also sidesteps DECA's TestData glob, which misses .jpeg.
from PIL import Image, ImageOps
import gc
exts, MAXSIDE, TMP = ('.jpg', '.jpeg', '.png', '.bmp'), 1024, '/content/_resized'
os.makedirs(TMP, exist_ok=True)
paths = []
for f in sorted(os.listdir(IN)):
    if os.path.splitext(f)[1].lower() not in exts: continue
    im = ImageOps.exif_transpose(Image.open(os.path.join(IN, f)).convert('RGB'))
    im.thumbnail((MAXSIDE, MAXSIDE))
    p = os.path.join(TMP, os.path.splitext(f)[0] + '.jpg'); im.save(p, quality=95); paths.append(p)

testdata = datasets.TestData(paths, iscrop=True, face_detector='fan')  # FAN detector crops to 224
for i in range(len(testdata)):
    with torch.no_grad():
        d = testdata[i]; name = d['imagename']
        images = d['image'].to(device)[None, ...]
        codedict = deca.encode(images)                                  # photo -> params (no renderer)
        verts, _, _ = deca.flame(shape_params=codedict['shape'],        # params -> mesh verts (no renderer)
                                 expression_params=codedict['exp'],
                                 pose_params=codedict['pose'])
    os.makedirs(f'{OUT}/{name}', exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[180, 180, 200, 255])  # flat gray "Skyrim"
    mesh.export(f'{OUT}/{name}/{name}.glb')
    params = {k: v.detach().cpu().numpy() for k, v in codedict.items() if torch.is_tensor(v) and k != 'images'}
    np.savez(f'{OUT}/{name}/{name}_params.npz', **params)               # your "JSON" — every key is a knob
    del images, codedict, verts; gc.collect(); torch.cuda.empty_cache()  # free GPU mem between images
    print('ok', name, '| shape', params['shape'].shape)
```

If you hit `AttributeError: 'DECA' object has no attribute 'render'`, something is still calling the renderer — we're calling `deca.flame(...)` directly to avoid exactly that, so check you didn't call `deca.decode(...)`.

### Cell 6 — the Skyrim-slider demo (mutate params, re-decode, no re-encode)

```python
import numpy as np, torch, trimesh

name = '<one of your imagenames>'
p = np.load(f'{OUT}/{name}/{name}_params.npz')
shape = torch.tensor(p['shape']).to(device)
exp   = torch.tensor(p['exp']).to(device)
pose  = torch.tensor(p['pose']).to(device)

# Tweak IDENTITY only; keep expression + pose so the face keeps its smile/angle.
shape = shape.clone()
shape[0, 0] *= -1.0     # flip the biggest shape PC (≈ face width)
shape[0, 1] *= 1.5      # exaggerate the next mode
# (later, Stage 2 replaces these hand-edits with a deterministic, key-seeded transform)

with torch.no_grad():
    verts, _, _ = deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
mesh = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), faces=faces, process=False)
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[200, 180, 180, 255])  # warm = tweaked
mesh.export(f'{OUT}/{name}/{name}_tweaked.glb')
print('exported tweaked glb')
```

Now you have `name.glb` (original) and `name_tweaked.glb` (mutated identity) in Drive — drop both into the viewer and A/B them.

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

- **Colab resets wipe everything installed.** Free-tier kills idle sessions (~90 min) and the runtime's filesystem is ephemeral. Re-run Cells 1–4 on a fresh session (~a few minutes). Your photos/outputs survive because they live in Drive. **You can no longer pin Colab to Python 3.10** (it aged out of Colab's 1-year runtime window); 3.11 is the oldest selectable and the choice doesn't persist across sessions, so just make the setup cells idempotent and re-runnable.
- **chumpy import errors** (`module 'inspect' has no attribute 'getargspec'`, or numpy `bool`/`str` AttributeErrors) mean Cell 4 didn't run before DECA was constructed (Cell 5). Run it first.
- **`TestData` silently finds 0 images** (run ends with no `ok …` prints and no error). DECA's directory glob only matches `*.jpg/*.png/*.bmp`, so `.jpeg` and uppercase extensions yield an empty set. Cell 5 avoids this by passing an explicit, case-insensitive file list rather than the directory path — keep that if you change input loading.
- **CUDA out of memory in the FAN detector** (OOM at `data[i]` → `conv2d`, trying to allocate many GiB). The face detector runs on the *full-resolution* image, and phone photos are big enough to blow the T4's ~14.5 GB. Cell 5 downscales each image to ≤1024 px before detection (DECA only needs ~224 px around the face) and frees GPU memory each iteration. If an OOM still leaves the GPU pinned (IPython's saved traceback holds the tensors), `Runtime → Restart`, then re-run setup + this cell.
- **Face detection quality.** DECA wants a clear, frontal-ish face; profile shots, sunglasses, hands/hair over the face, and tiny faces degrade it. EXIF rotation on iPhone photos is a common silent failure — if a detect fails, re-export the photo (bakes in rotation) or strip EXIF. One face per image (the demo reconstructs one).
- **"It doesn't really look like me."** Expected, not a bug — single-image FLAME regression is constrained to FLAME's learned face manifold (generic noses, puffy chins). Recognizable but stylized. MICA gives more metric identity if this ever matters; don't switch yet.
- **`deca.flame(...)` kwargs / `faces_tensor`** (verified on the current DECA build; portability note for other revisions). If `faces_tensor` is missing, load faces from `deca_cfg.model.topology_path` (the bundled `head_template.obj`). If the `flame()` kwargs differ, check `decalib/models/FLAME.py`'s `forward` signature.

---

## 6. What to do when it works

### Sanity check the identity vector
Run Cell 5 on three photos of the same person and two of someone else; compare `shape` vectors:

```python
import numpy as np
load = lambda n: np.load(f'{OUT}/{n}/{n}_params.npz')['shape'][0]
a, b = load('personA_1'), load('personA_2')
c    = load('personB_1')
print("same person:", np.linalg.norm(a - b))
print("different people:", np.linalg.norm(a - c))   # should be visibly larger
```

This is your first intuition for what the identity vector encodes.

### The slider intuition (Cell 6)
Flip/scale individual `shape` coefficients and re-view. The first few are the loud perceptual modes (face width, head length, nose prominence-ish); later ones are finer. Knowing which dims are loud vs. quiet is exactly what tells you what a Stage-2 hash should mutate hard vs. leave alone.

### Next milestones
1. ✅ Photo → params + mesh → viewer (this guide).
2. ✅ Manually tweak params and re-render (Cell 6).
3. **Stage 2 — the hash.** Replace the hand-edits with `transform(params, key) -> params'` behind a hot-swappable strategy interface (start: seeded Gaussian offset on `shape`, clamped to ±2σ; preserve `exp`/`pose`). See the research report.
4. **Stage 4 — photorealism.** Layer Arc2Face / InstantID / PuLID once the 3D pipeline feels good.

---

## 7. Keeping the notebook, these docs, and Claude in sync

The hard part of this project's workflow: the *live* notebook lives at a Colab URL (`colab/DECA.ipynb - Colab.webloc`), the repo has no canonical `.ipynb`, and Claude edits files in the repo — so the notebook and the docs/Claude drift apart easily. Three strategies, increasing robustness:

1. **Markdown mirror (lowest effort, what this doc is).** §3's cells are the canonical text. When you change a cell in Colab, mirror it here (or vice-versa). Simple, but Claude never sees the *actual* notebook — you paste cell output when something breaks.
2. **Version the notebook in the repo (recommended next step).** In Colab: **File → Save a copy in GitHub**, committing the `.ipynb` to this repo. Then Claude can read the real notebook and propose exact cell edits, and you have history. Notebooks are code, not gitignored data, so this is safe (your photos/weights stay in Drive). Re-save after meaningful changes.
3. **Logic in a versioned `.py`, notebook as a thin runner (best; `pipeline.py` already exists for this).** The repo's `pipeline.py` folds Cells 1–4 into `bootstrap()` and Cells 5–6 into `load_deca()` / `reconstruct()` / `tweak()`. The whole notebook collapses to:

   ```python
   !git clone https://github.com/tragicallyludicrous/face-hashing.git
   import sys; sys.path.insert(0, "face-hashing")
   import pipeline
   pipeline.bootstrap()                   # clone DECA, install, patch, cache weights
   deca, faces = pipeline.load_deca()     # renderer-free DECA + FLAME topology
   pipeline.reconstruct(deca, faces)      # Input/ -> Output/<name>/{.glb,_params.npz}
   pipeline.tweak(deca, faces, "<name>")  # mutate identity, export <name>_tweaked.glb
   ```

   Claude authors `pipeline.py` directly (clean diffs, no notebook-JSON noise); the notebook barely changes; data still flows through Drive. It's also the natural home for the Stage-2 transform (extend `default_mutation` / add a strategy registry). To pick up edits in a later session, `!git -C face-hashing pull` (or re-clone). The same portability fallbacks from §3 are in `pipeline.py` too.

**Recommendation:** adopt #2 now (so the notebook is versioned and Claude can see ground truth), and migrate to #3 as the pipeline stabilizes. Keep this guide (#1) as the human-readable explanation regardless.
