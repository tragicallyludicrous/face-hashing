# Face Hashing — Project Setup Guide (Stage 1)

**Goal of this milestone:** Upload a photo, run DECA on Google Colab, get back FLAME parameters (your "JSON object" of facial features) and a 3D mesh, view that mesh by drag-rotating in your browser.

**No transform yet.** That's intentional. Stage 1 answers one question: *can we reliably get from a photo to a structured 3D representation we can play with?*

---

## How this guide is organized

1. **The mental model** — what the pieces are and why
2. **Repo layout** — where everything lives
3. **Step-by-step setup** — Colab notebook + local viewer
4. **Known footguns** — the things that will trip you up
5. **What to do when it works** — sanity checks and next moves

If you've never used Colab before, skim section 1 first. If something breaks, jump to section 4 — your problem is probably listed there.

---

## 1. The mental model

You're building a two-machine system:

**Machine A: Google Colab (the GPU)**
Runs the heavy ML stuff. You upload a photo here, DECA chews on it, and out come two things:
- A FLAME parameter dict — Python dict of tensors, the "facial features as JSON" representation
- A 3D mesh — a `.obj` file you'll convert to `.glb`

**Machine B: Your Mac (the viewer)**
Runs a tiny static HTML page that loads the `.glb` and lets you drag-rotate it. No build step, no framework — just one HTML file.

You download the artifacts from Colab to your Mac after each run. Later, when you wire up the full pipeline, the "two machines" will become a backend (Colab/RunPod/local) and a frontend (your web app), but for now keep them mentally separate.

### Glossary you'll bump into

- **FLAME** — A parametric 3D face model. Given a vector of ~300 shape coefficients + ~100 expression coefficients + 6 pose values, it spits out a 5023-vertex 3D mesh of a face. Created by the Max Planck Institute. Think of it as the underlying "rig" that every individual face is a setting of.
- **DECA** — A neural network that takes a 2D photo and predicts the FLAME parameters for the face in that photo. This is the regressor — the "photo → JSON" step.
- **`.obj`** — Plain-text 3D mesh format. Easy to read, terrible for the browser.
- **`.glb`** — Binary glTF. The format `<model-viewer>` wants. We convert `.obj` → `.glb` so the browser is happy.
- **`<model-viewer>`** — A Google-maintained web component. Drop one HTML tag, get a drag-rotatable 3D viewer with proper lighting. No three.js code required.

---

## 2. Repo layout

Create this folder structure on your Mac. Empty for now; we'll fill it as we go.

```
face-hashing/
├── README.md
├── colab/
│   └── stage1_deca.ipynb          # The notebook (we'll build it in Colab and save here)
├── outputs/                        # Where downloaded results go
│   └── .gitkeep
├── viewer/
│   ├── index.html                  # The <model-viewer> page
│   └── models/                     # .glb files go here
└── .gitignore
```

**`.gitignore`** — start with this:

```
outputs/*
!outputs/.gitkeep
viewer/models/*.glb
.DS_Store
__pycache__/
*.pyc
.venv/
```

You will eventually be tempted to check in face data and model weights. Don't. The weights are large and have license restrictions; the photos are personal data. Keep them local.

**Initialize the repo:**

```bash
mkdir -p face-hashing/{colab,outputs,viewer/models}
cd face-hashing
touch outputs/.gitkeep
git init
```

---

## 3. Step-by-step setup

### 3.1 — Register for FLAME (do this first, takes ~5 minutes)

DECA depends on the FLAME model files, which require a free registration.

1. Go to https://flame.is.tue.mpg.de/register.php
2. Register with your email (real name and institution are fine; "Personal" is an acceptable institution).
3. Wait for the confirmation email. Click the link.
4. Log in at https://flame.is.tue.mpg.de/login.php
5. Don't download anything yet — you'll grab the files from inside Colab using your credentials.

**Licensing note:** FLAME 2020 is under a Creative Commons Attribution license (commercial use allowed with attribution, no fake/defamatory content, no pornography). The newer FLAME 2023 Open is CC-BY-4.0. DECA uses FLAME 2020 by default; that's fine for your project.

### 3.2 — Open Google Colab

1. Go to https://colab.research.google.com
2. Sign in with a Google account
3. **File → New notebook**
4. **Runtime → Change runtime type → T4 GPU → Save**

You now have a free Linux box with an NVIDIA T4 GPU for up to ~12 hours. If you let it idle for 90 minutes it'll disconnect and you'll lose installed packages (but not files saved to Google Drive). For Stage 1 this is fine.

**Verify the GPU is attached.** In a cell:

```python
!nvidia-smi
```

You should see a table mentioning "Tesla T4." If you see "command not found," the runtime type didn't take — recheck step 4 above.

### 3.3 — Install DECA in Colab

The DECA repo (https://github.com/yfeng95/DECA) hasn't been updated since 2021 and its `requirements.txt` pins old versions. We're going to install it but override the troublesome pins. Walk through these cells one at a time.

**Cell 1 — clone the repo:**

```python
!git clone https://github.com/yfeng95/DECA.git
%cd DECA
```

**Cell 2 — install Python dependencies.** DECA's own `install_conda.sh` is meant for a conda environment; on Colab we install with pip and skip the parts that conflict with Colab's preinstalled libraries:

```python
# Colab already has a recent PyTorch + CUDA, leave those alone.
!pip install -q chumpy
!pip install -q yacs==0.1.8
!pip install -q face-alignment
!pip install -q ninja
!pip install -q kornia==0.6.12
!pip install -q scikit-image
!pip install -q opencv-python
!pip install -q PyYAML
```

**Cell 3 — chumpy needs a Python 3.10+ patch.** This is the single most common DECA install failure. chumpy uses `numpy.bool` which was removed; patch it:

```python
import chumpy.ch
import numpy as np
# Monkey-patch deprecated numpy aliases that chumpy still uses
for alias, real in [('bool', bool), ('int', int), ('float', float), ('complex', complex), ('object', object), ('str', str)]:
    if not hasattr(np, alias):
        setattr(np, alias, real)
```

(If you're on Python <3.10 in Colab, this cell is a no-op. Run it anyway.)

**Cell 4 — install pytorch3d (the renderer).** DECA's *default* rasterizer uses PyTorch JIT compilation, which sometimes fails on Colab. The safer path is to install PyTorch3D and pass `--rasterizer_type=pytorch3d` later. Use the prebuilt wheel matching Colab's PyTorch:

```python
import torch
print(torch.__version__, torch.version.cuda)

# Install prebuilt pytorch3d (this avoids a 20-minute source build)
!pip install -q fvcore iopath
!pip install -q --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
```

If the prebuilt wheel URL fails because Colab's PyTorch/CUDA combo changed, fall back to:

```python
!pip install -q "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

This builds from source and takes ~10–20 minutes but works on any combo. Have coffee.

### 3.4 — Download the model weights

DECA needs three files. Two are from FLAME (the registration you did in step 3.1), one is DECA's own weights.

**Cell 5 — set up directories:**

```python
import os
os.makedirs('data', exist_ok=True)
```

**Cell 6 — get DECA's pretrained weights.** These live on Google Drive. The DECA repo's README links to a specific file `deca_model.tar`:

```python
!pip install -q gdown
# DECA pretrained model (~430MB)
!gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O data/deca_model.tar
```

If `gdown` fails (Google sometimes throttles), you can manually:
1. Go to the DECA README on GitHub
2. Click the "trained model" link, download `deca_model.tar`
3. Upload it to Colab via the file browser (left sidebar → folder icon → upload to `DECA/data/`)

**Cell 7 — get FLAME 2020.** This is the file that requires registration. The cleanest way in Colab is to upload it manually:

1. On your laptop, log in at https://flame.is.tue.mpg.de
2. Go to **Downloads → FLAME 2020 → "Model and Code"** and download `FLAME2020.zip`
3. In Colab, click the folder icon in the left sidebar, then navigate into `DECA/data/`, right-click → **Upload**, choose the zip
4. Then in a cell:

```python
!cd data && unzip -o FLAME2020.zip
!ls data
```

You should see `generic_model.pkl` among the contents. That's the FLAME 2020 model file DECA needs.

**Cell 8 — get the FLAME albedo and other auxiliary files.** DECA's README lists these under "Prepare data":

```python
# FLAME albedo model (also from MPI) — optional, only needed for textured output
# Skip for now; we're going with untextured mesh first.

# DECA's own auxiliary files (already in the repo at DECA/data/)
!ls data
```

You should see `deca_model.tar`, `generic_model.pkl`, and the DECA repo's bundled files (`fixed_displacement_256.npy`, `landmark_embedding.npy`, `mean_texture.jpg`, `texture_data_256.npy`, `uv_face_eye_mask.png`, `uv_face_mask.png`). If `generic_model.pkl` is missing, re-do Cell 7.

### 3.5 — Run DECA on a test image

**Cell 9 — upload a photo.** Use one of DECA's bundled examples first to confirm the pipeline works, then swap in your own:

```python
# Use a bundled example
!ls TestSamples/examples
```

Pick one, e.g., `alfw2000.jpg`. To use your own photo, upload it via the sidebar file browser into a folder like `DECA/my_inputs/`.

**Cell 10 — run reconstruction:**

```python
!python demos/demo_reconstruct.py \
    -i TestSamples/examples \
    -s outputs/examples \
    --saveObj True \
    --saveMat True \
    --saveVis True \
    --rasterizer_type=pytorch3d
```

Flags explained:
- `-i` input folder (DECA processes every image in it)
- `-s` save folder
- `--saveObj True` exports the 3D mesh
- `--saveMat True` exports a `.mat` file with the FLAME parameters — **this is your "JSON"**
- `--saveVis True` exports a visualization image (input, landmarks, reconstructed face overlay)
- `--rasterizer_type=pytorch3d` uses the safer renderer

If it finishes without errors you'll have `outputs/examples/<imagename>/<imagename>.obj` and `<imagename>.mat`.

### 3.6 — Inspect the FLAME parameters (the "JSON")

The `.mat` file is MATLAB format. Load it into Python so you can see what's actually in there:

**Cell 11:**

```python
from scipy.io import loadmat
import numpy as np

mat = loadmat('outputs/examples/alfw2000/alfw2000.mat')
# Drop MATLAB's internal keys
params = {k: v for k, v in mat.items() if not k.startswith('__')}

for key, val in params.items():
    print(f"{key:20s} shape={val.shape} dtype={val.dtype}")
```

You should see entries like:

```
shape                shape=(1, 100)   dtype=float32   ← identity (100 coefficients)
exp                  shape=(1, 50)    dtype=float32   ← expression (50 coefficients)
pose                 shape=(1, 6)     dtype=float32   ← jaw + global rotation
cam                  shape=(1, 3)     dtype=float32   ← camera (scale + 2D offset)
light                shape=(1, 9, 3)  dtype=float32   ← spherical-harmonics lighting
tex                  shape=(1, 50)    dtype=float32   ← texture (50 coefficients)
detail               shape=(1, 128)   dtype=float32   ← person-specific wrinkles
```

**This is the structured representation your Face Hashing project is built around.** Every single number is a knob. `shape[0][0]` through `shape[0][99]` control identity — those are the ones your eventual hash function will mutate. `exp` and `pose` you'll probably *preserve* so the hashed face inherits the original's smile and head angle.

Save it as actual JSON for the warm fuzzies:

**Cell 12:**

```python
import json

params_json = {k: v.tolist() for k, v in params.items()}
with open('outputs/examples/alfw2000/params.json', 'w') as f:
    json.dump(params_json, f, indent=2)

print("Saved params.json")
# Peek at the first 5 shape coefficients
print("First 5 identity coefficients:", params['shape'][0][:5])
```

### 3.7 — Convert the mesh to `.glb` for the browser viewer

DECA outputs `.obj`. `<model-viewer>` needs `.glb`. The cleanest converter is `trimesh`:

**Cell 13:**

```python
!pip install -q trimesh
```

**Cell 14:**

```python
import trimesh

mesh = trimesh.load('outputs/examples/alfw2000/alfw2000.obj', force='mesh')
# Strip texture references (we're going untextured for the Skyrim look)
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[180, 180, 200, 255])
mesh.export('outputs/examples/alfw2000/alfw2000.glb')
print("Exported .glb")
```

The vertex color `[180, 180, 200, 255]` gives you a slightly cool gray — close to the Skyrim character-creator preview aesthetic. Adjust to taste.

### 3.8 — Download the artifacts

**Cell 15:**

```python
from google.colab import files
files.download('outputs/examples/alfw2000/alfw2000.glb')
files.download('outputs/examples/alfw2000/params.json')
```

Or zip the whole output folder:

```python
!cd outputs && zip -r alfw2000_bundle.zip examples/alfw2000
files.download('outputs/alfw2000_bundle.zip')
```

Move the `.glb` into your `face-hashing/viewer/models/` folder on your Mac. Save `params.json` to `face-hashing/outputs/`.

### 3.9 — The viewer (on your Mac)

Create `face-hashing/viewer/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Face Hashing — Stage 1 viewer</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    main { max-width: 900px; margin: 2rem auto; padding: 1rem; }
    model-viewer {
      width: 100%;
      height: 70vh;
      background: linear-gradient(180deg, #222, #111);
      --poster-color: transparent;
    }
    h1 { font-weight: 300; }
    label { display: block; margin: 1rem 0 0.5rem; }
    select { padding: 0.4rem; background: #222; color: #eee; border: 1px solid #444; }
  </style>
</head>
<body>
  <main>
    <h1>Face Hashing — Stage 1</h1>
    <label for="model-select">Model:</label>
    <select id="model-select">
      <option value="models/alfw2000.glb">alfw2000 (example)</option>
    </select>
    <model-viewer
      id="viewer"
      src="models/alfw2000.glb"
      camera-controls
      auto-rotate
      auto-rotate-delay="3000"
      shadow-intensity="1"
      exposure="1.1"
      camera-orbit="0deg 75deg 0.5m"
      field-of-view="30deg"
      alt="Reconstructed 3D face"
    ></model-viewer>
  </main>
  <script>
    const sel = document.getElementById('model-select');
    const viewer = document.getElementById('viewer');
    sel.addEventListener('change', e => { viewer.src = e.target.value; });
  </script>
</body>
</html>
```

**Run it.** Browsers won't load local files via `file://` for security reasons, so serve it:

```bash
cd face-hashing/viewer
python3 -m http.server 8080
```

Open http://localhost:8080. You should see the face. Drag to rotate, scroll to zoom.

**That's Stage 1 complete.** Photo → FLAME params (JSON) → 3D mesh → interactive viewer.

---

## 4. Known footguns

These are the things that will eat your afternoon. Read them now, refer back when something breaks.

### 4.1 — Colab disconnects and you lose everything
Free-tier Colab kills idle sessions after ~90 minutes and total sessions after ~12 hours. When this happens, your installed packages are gone but your files might be too (the runtime's filesystem is ephemeral).

**Fix:** Mount Google Drive and save outputs there:

```python
from google.colab import drive
drive.mount('/content/drive')
# Then save to /content/drive/MyDrive/face-hashing/outputs/
```

For Stage 1 it's not worth bothering — re-running the whole notebook on a fresh session takes ~10 minutes. But once you start iterating, mount Drive.

### 4.2 — `chumpy` import fails with `AttributeError: module 'numpy' has no attribute 'bool'`
chumpy is unmaintained and uses numpy aliases that were removed in numpy 1.20+. The patch in Cell 3 handles it. If you see this error, you skipped or moved Cell 3 — run the patch *before* any DECA import.

### 4.3 — `pytorch3d` install hangs or fails
PyTorch3D wheels are pinned to specific PyTorch + CUDA versions. Colab updates its PyTorch periodically, breaking the wheel URL.

**Diagnosis:**
```python
import torch
print(torch.__version__, torch.version.cuda)
```
Match this against https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md to find the right wheel URL.

**Last resort:** the `git+https://...@stable` install in Cell 4. It always works, just takes 10–20 minutes.

### 4.4 — `RuntimeError: Error(s) in loading state_dict for DECA`
Usually means `deca_model.tar` didn't download fully, or `generic_model.pkl` is missing.

```python
import os
print(os.path.getsize('data/deca_model.tar'))  # expect ~430 MB
print(os.path.exists('data/generic_model.pkl'))  # expect True
```

### 4.5 — `<model-viewer>` shows nothing / blank canvas
Three usual causes:
1. **Wrong path** — `src="models/alfw2000.glb"` is relative to the HTML file. Make sure the `.glb` is actually in `viewer/models/`.
2. **You opened the HTML via `file://`** — won't work, browsers block this. Use the `python3 -m http.server` trick.
3. **The mesh is too small or too far away** — DECA meshes are in a small coordinate range. The `camera-orbit="0deg 75deg 0.5m"` attribute in the HTML handles this; if you're using your own viewer, set the camera distance to ~0.5 meters.

### 4.6 — Face is detected wrong or output looks weird
DECA needs a clear, frontal-ish face. Profile shots, heavy occlusion (sunglasses, hands), extreme expressions, and very low resolution all hurt. Crop your input photo to roughly square around the face before uploading. The `--useTex True` flag also affects this slightly; leave it off for now.

### 4.7 — "Why does the 3D face not really look like me?"
This is **not a bug** — it's a fundamental limitation of single-image 3DMM regression. DECA captures identity-relevant geometry but the mesh is constrained to FLAME's learned face manifold, which biases toward average proportions. Noses come out generic, chins puffy. Recognizable but stylized. If this bothers you later, MICA (https://github.com/Zielon/MICA) gives metrically accurate identity reconstruction at the cost of expression fidelity. Don't switch yet — get the full pipeline working first.

### 4.8 — The `.mat` file from DECA stores nested dicts oddly
`scipy.io.loadmat` adds keys like `__header__`, `__version__`, `__globals__` — filter them out (Cell 11 already does this). If you load the file and see weird `mat_struct` objects, pass `squeeze_me=True, struct_as_record=False` to `loadmat`.

### 4.9 — License / commercial use
DECA's code is MIT but its pretrained models are non-commercial (per the DECA repo: research use only). FLAME 2020 is CC-BY (commercial OK with attribution and the listed restrictions). For your personal-project / exploratory phase, you're fine. If this ever turns into a product, you'd need either to retrain DECA-equivalent regressor on a permissive dataset, or get a license from MPI.

### 4.10 — You'll want to compare two outputs and the viewer only loads one
The `index.html` already has a `<select>` — duplicate the option line for each new `.glb` you drop into `viewer/models/`. Later we'll automate this with a small script that lists the directory; not worth it yet.

---

## 5. What to do when it works

### 5.1 — Sanity checks

Run DECA on three different photos of the same person and compare the `shape` coefficients in their `params.json` files. They should be close but not identical (lighting and angle introduce noise). Run on photos of two different people; the `shape` vectors should be visibly different. This is your first hands-on intuition for what the "identity vector" actually encodes.

```python
import numpy as np, json
a = json.load(open('outputs/personA_photo1/params.json'))
b = json.load(open('outputs/personA_photo2/params.json'))
c = json.load(open('outputs/personB_photo1/params.json'))

sa, sb, sc = np.array(a['shape'])[0], np.array(b['shape'])[0], np.array(c['shape'])[0]
print("Same person, different photos:", np.linalg.norm(sa - sb))
print("Different people:", np.linalg.norm(sa - sc))
```

The second number should be visibly larger than the first. If it's not, your photos are probably very different in lighting/pose; try frontal photos with neutral expression.

### 5.2 — Manual parameter tweaking (your first taste of "the transform")

This is the fun part. Don't write a transform yet; just *manually* change a few numbers and re-render.

In Colab, after running DECA once, you can re-render with modified parameters by calling DECA's decoder directly. Easiest path: write a small script that loads a `.mat`, mutates `params['shape'][0][0] *= -1`, and passes the modified params back through FLAME's PyTorch layer to get a new `.obj`. The DECA repo has examples in `demos/demo_transfer.py` that do something similar (transferring expressions between faces). Read that script — it shows you exactly how to call FLAME with custom parameters.

You'll quickly find that flipping the sign of `shape[0][0]` makes the face wider or narrower (it's the largest principal component of head shape variation). Coefficients 1–10 each correspond to a major mode of variation; later ones encode finer detail. There's no fixed semantic label per dimension — they're learned — but with a few minutes of poking you'll have rough names for the first few ("face width," "head length," "nose prominence," etc.).

**This is exactly the Skyrim-slider experience you described.** And it's the right intuition to have *before* designing a hash function: knowing which dimensions are perceptually loud vs quiet tells you what your transform should mutate hard vs leave alone.

### 5.3 — Next milestones (in rough order)

1. **Get FLAME-only re-rendering working** — load params, modify, render new `.obj`, view it. Closes the loop on "I can manipulate the JSON and see the result."
2. **Build a tiny parameter-editor UI** — extend the HTML viewer with a sidebar of sliders that maps to a few shape coefficients. Each slider edit triggers a re-render (you can do this client-side with three.js + FLAME-in-JS, or by round-tripping to a small Python backend).
3. **Stage 2: the actual hash function** — design and plug in `transform(params, key) -> params'`. Start simple: seeded Gaussian offset on shape coefficients only.
4. **Stage 4: photorealistic** — once the 3D pipeline works, layer in Arc2Face or InstantID to make the output a photo instead of a mesh. The mesh is great for understanding; a photorealistic result is great for the demo.

---

## 6. If you get stuck

The first place to look is the DECA repo's **Issues** tab on GitHub — most install problems have been hit and resolved there. Search for your exact error message. Second, the EMOCA / INFERNO repos (https://github.com/radekd91/inferno) have more recent install instructions and sometimes their fixes apply to DECA.

If you hit a wall, paste the full error traceback and tell me which cell failed. Most DECA issues are environmental (Python version, CUDA version, missing file) and quick to debug once we see the message.

Good luck. The first time you see a 3D face spin in your browser that came from a photo you uploaded ten minutes ago, it's a great feeling.
