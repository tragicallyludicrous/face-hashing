# Running MICA Stage 1 locally on the Mac (no Colab, no CUDA)

Goal: prove that MICA identity extraction — photo → 300-d `identity.npy` + neutral mesh — runs
**off-Colab on Apple Silicon**, so the local-first plan in `../stage2-design.md` (§7) is viable.

**Why this is possible:** MICA's whole *inference* path is PyTorch3D-free (verified by reading the
source — `demo.py → micalib/models/mica.py → micalib/base_model.py → models/{arcface,generator,flame}.py`).
PyTorch3D is only in the training/rendering code the demo never imports. So there's no renderer to
build; the port is just CPU-safety + the two unmaintained-dependency patches we already use in Colab.
`patch_mica_for_mac.py` applies all of it idempotently.

> **Status: not yet verified on hardware.** This is the port, structured to need at most one
> debugging pass on your machine. Run it, and if a step trips, see Troubleshooting below.

---

## 0. Prereqs

- Xcode command-line tools (`xcode-select --install`) — `insightface` builds a small C extension.
- **Python 3.11 (arm64).** NOT 3.13/3.14 — this stack (torch, insightface, onnxruntime, chumpy)
  has no wheels there and chumpy's sdist won't build. On this Mac the right interpreter is
  `/usr/local/bin/python3.11` (verified arm64). Check yours: `python3.11 -c "import platform;
  print(platform.machine())"` must print `arm64` (x86_64 = Rosetta, loses MPS — don't use it).
- The two weight files you already have in the Drive alias:
  - `…/Face-Hashing/FLAME/FLAME2020/generic_model.pkl`  (FLAME 2020, ~50 MB)
  - `…/Face-Hashing/cache/mica.tar`  (MICA checkpoint, ~480 MB)
- A test photo or two (use one you also ran in Colab, so you can cross-check the output).

## 1. Environment

```bash
cd "<repo>/local"
/usr/local/bin/python3.11 -m venv .venv && source .venv/bin/activate   # 3.11 arm64, NOT the 3.14 default
pip install -U pip setuptools wheel
# CPU/MPS torch (default macOS wheels are already MPS-capable), + MICA's runtime deps
pip install torch torchvision \
            insightface onnxruntime \
            trimesh loguru yacs opencv-python scikit-image numpy gdown tqdm \
            face-alignment
# chumpy last + WITHOUT build isolation: its setup.py does `import pip`, which fails under
# pip's isolated build env. --no-build-isolation runs the build in this venv (pip present).
pip install --no-build-isolation chumpy
```

## 2. Clone MICA and patch it

```bash
git clone https://github.com/Zielon/MICA.git
python patch_mica_for_mac.py MICA      # chumpy shim + LandmarksType + demo.py CPU/MPS-safety
```

## 3. Drop in the weights (the repo already ships `landmark_embedding.npy`)

```bash
DRIVE="$HOME/<path-to>/My Drive/Face-Hashing"   # your Drive alias
mkdir -p MICA/data/FLAME2020 MICA/data/pretrained
cp "$DRIVE/FLAME/FLAME2020/generic_model.pkl" MICA/data/FLAME2020/
cp "$DRIVE/cache/mica.tar"                     MICA/data/pretrained/
```

## 3b. Detector pack (antelopev2, ~360 MB, once)

MICA's `landmark_detector.py` uses InsightFace's `antelopev2` pack for face detection/cropping;
it isn't auto-downloaded, and a fresh `~/.insightface` makes `FaceAnalysis` fail with
`AssertionError: 'detection' in self.models`. Fetch + flatten it once (same fix as Colab M4 — the
zip nests as `antelopev2/antelopev2/*.onnx`):

```bash
python - <<'PY'
import os, glob, shutil
from insightface.utils.storage import ensure_available
base = os.path.expanduser('~/.insightface/models/antelopev2')
if not glob.glob(base + '/**/*.onnx', recursive=True):
    ensure_available('models', 'antelopev2', root=os.path.expanduser('~/.insightface'))
os.makedirs(base, exist_ok=True)
moved = 0
for f in glob.glob(base + '/**/*.onnx', recursive=True):
    dst = os.path.join(base, os.path.basename(f))
    if os.path.abspath(f) != os.path.abspath(dst):
        shutil.move(f, dst); moved += 1
print('antelopev2 ready; moved', moved, '; files:', sorted(os.listdir(base)))
PY
```

(If `ensure_available` 404s, the HuggingFace mirror fallback from Colab M4 still applies.)

## 4. Run it — **force CPU for the first proof**

```bash
mkdir -p in arcface_tmp out
cp /path/to/one_test_photo.jpg in/

cd MICA
MICA_DEVICE=cpu python demo.py \
    -i ../in -o ../out -a ../arcface_tmp -m data/pretrained/mica.tar
```

Expected: `out/<photo-stem>/` containing `identity.npy` (300-d), `mesh.ply`, `mesh.obj`, `kpt68.npy`.
(First run also downloads InsightFace's RetinaFace detector to `~/.insightface` — needs network.)

## 5. Verify it actually worked (the real proof)

The local run writes the raw **`identity.npy`** (300-d) per photo. Colab never synced a `.npy` —
that file only existed on the ephemeral runtime; its export cell repackaged the identity as
**`<stem>_mica_identity.npz`** (key `shape`) so it would land in Drive. So compare the local
`identity.npy` against Drive's `_mica_identity.npz`.

> Don't grab `<stem>_params.npz` by mistake — that's **DECA** (100-d `shape`), a different model.
> The MICA identity is `<stem>_mica_identity.npz` (300-d).

**Single photo:**

```python
import numpy as np, os
stem      = "emma-restaurant"                              # a photo run in BOTH places
LOCAL_OUT = "out"                                          # = local/out  (run this from local/)
DRIVE_OUT = os.path.expanduser("~/<your-path>/My Drive/Face-Hashing/Output")

local = np.load(f"{LOCAL_OUT}/{stem}/identity.npy").ravel()
colab = np.load(f"{DRIVE_OUT}/{stem}/{stem}_mica_identity.npz")["shape"].ravel()
print(stem, "dims", local.shape, colab.shape,
      "cos", float(local @ colab / (np.linalg.norm(local) * np.linalg.norm(colab))))
```

**All photos at once:**

```python
import numpy as np, os, glob
LOCAL_OUT = "out"
DRIVE_OUT = os.path.expanduser("~/<your-path>/My Drive/Face-Hashing/Output")
for p in sorted(glob.glob(f"{LOCAL_OUT}/*/identity.npy")):
    stem = os.path.basename(os.path.dirname(p))
    cz = f"{DRIVE_OUT}/{stem}/{stem}_mica_identity.npz"
    if not os.path.exists(cz):
        print(f"{stem:28} (no Colab npz to compare)"); continue
    a = np.load(p).ravel(); b = np.load(cz)["shape"].ravel()
    print(f"{stem:28} cos={float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))):.4f}  "
          f"L2={np.linalg.norm(a - b):.3f}")
```

Same weights + deterministic model → cosine should be **≳ 0.999** (tiny CPU-vs-CUDA float drift
only). If so, **Stage 1 runs off-Colab** and the local-first pivot (`../stage2-design.md` §7) is green.

## 6. (Optional) try MPS for speed

Once CPU is confirmed correct:

```bash
MICA_DEVICE=mps python demo.py -i ../in -o ../out -a ../arcface_tmp -m data/pretrained/mica.tar
```

The patch sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (unimplemented ops fall back to CPU with a warning)
and casts the input tensors to float32 (MPS has no float64 — `imread()/255.` is float64). If MPS
still throws a device-placement error (some FLAME buffers may stay on CPU), don't fight it — CPU is
already confirmed faithful (cosine 1.0 vs Colab) and is fine for enrollment (once per user) and
one-off runs. MPS is a nice-to-have, not the deliverable.

---

## Troubleshooting

- **`ModuleNotFoundError: pytorch3d`** — shouldn't happen (inference path is clean), but if some
  transitively-imported module surprises us, it's import-only, not used: `mkdir -p
  MICA/pytorch3d_stub/pytorch3d && touch MICA/pytorch3d_stub/pytorch3d/__init__.py` and prepend
  `PYTHONPATH=pytorch3d_stub`. Tell me which module pulled it and I'll fold a proper stub into the patch.
- **chumpy fails to *build*** (`ModuleNotFoundError: No module named 'pip'` while "Getting requirements
  to build wheel") — its setup.py imports pip, unavailable under build isolation. Install it with
  `pip install --no-build-isolation chumpy` (see §1). If you see this, you're almost certainly also on
  the wrong Python — confirm `python --version` inside the venv says 3.11, not 3.14.
- **chumpy errors at FLAME *load*** (`getargspec` / `numpy has no attribute 'bool'`) — the shim didn't
  apply. Re-run `patch_mica_for_mac.py MICA` *after* `pip install chumpy` (it patches chumpy in place).
- **`FileNotFoundError: …/landmark_embedding.npy`** — the clone didn't include it. Copy it from your
  DECA/FLAME data into `MICA/data/FLAME2020/landmark_embedding.npy`.
- **`Face not detected`** — RetinaFace missed the face (small/sideways). Use a clearer, more frontal
  test photo for the first run.
- **`insightface` build fails** — ensure Xcode CLT is installed; on Apple Silicon `pip install
  insightface` compiles from source.
- **`torch.cuda.amp.autocast` / device errors** — the demo patch neutralizes autocast and routes
  everything through `_DEVICE`; if you see a stray `.cuda` the regex missed, paste the traceback.

## The clean way to run — `mica_local.py`

`demo.py` (§4) is the *verification reference*. For everyday use there's now a clean, in-process
runner — **`mica_local.py`** — that drives MICA's same detector + network + FLAME decoder without the
subprocess or the ArcFace-blob disk round-trip `demo.py` does. From `local/`:

```bash
python mica_local.py -i in -o out                 # CPU -> out/<stem>/{identity.npy, arcface.npy, <stem>.glb}
python mica_local.py -i in -o out --device mps     # or MPS
```

`identity.npy` is the 300-d FLAME identity (the hash payload); `arcface.npy` is the 512-d ArcFace
embedding MICA consumes (the recognition "ceiling"). Inspect either with
`python ../tools/present.py --compare --source mica|arcface`.

Or as a library — this is Stage 1's local entry point (Stage 2 enrollment averages several `embed`
calls per person, per `../stage2-design.md` §4):

```python
import mica_local as mica
h   = mica.load(device="cpu")          # build detector + model ONCE
vec = mica.embed(h, "photo.jpg")       # (300,) identity, or None if no face
```

It assumes `patch_mica_for_mac.py` has been run against the checkout and the weights are in place
(§3/§3b). The on-disk source fixes come from the patcher; the runtime shims (CUDA-less autocast,
CPU-mapped `torch.load`, MPS env, device) are baked into `mica_local.py` itself, so it doesn't need
`demo.py`'s injected preamble.

**Verify it once against the reference:** run `demo.py` and `mica_local.py` on the same photo and
compare their `identity.npy` — cosine should be ~1.0 (same model, same preprocessing). After that,
prefer `mica_local.py`.

## Stage 3 — pose the identity: `compose_flame.py` (DECA) / `smirk_local.py` (SMIRK)

MICA gives a *neutral* identity. To put that identity into the original photo's expression and head
pose — the Stage-3 "reconstruct" — decode FLAME with **MICA shape + an expression/pose source**.
Two sources, same shared FLAME 2020 basis:

- **DECA** (from Colab `<stem>_params.npz`) — `compose_flame.py`:
  ```bash
  python compose_flame.py out/MICA/<stem>/identity.npy <…>/<stem>_params.npz -o out/<stem>/composed.glb
  ```
- **SMIRK** (CVPR'24, MIT — much better expressions, plus jaw + eyelids) — `smirk_local.py`, the
  renderer-free, in-process sibling of `mica_local.py` (same trick: import the encoder + FLAME, skip
  the PyTorch3D Renderer). Batch by default; pass one photo for a single compose:
  ```bash
  python smirk_local.py -i in -o out/SMIRK --mica out/MICA     # per stem: smirk_params.npz + smirk.glb (+composed.glb)
  python smirk_local.py in/<photo>.jpeg --mica out/MICA/<stem>/identity.npy -o out/<stem>/smirk.glb
  ```
  `smirk_params.npz` holds the per-photo payload (`shape/expression/pose/jaw/eyelid/cam`); `smirk.glb`
  is SMIRK's own native reconstruction; `composed.glb` is the MICA identity wearing this photo's pose.
  Sanity check: composing a raw MICA identity against its *own* photo barely moves the mesh (same
  person); a hashed identity diverges. SETUP is in the `smirk_local.py` docstring (clone
  `georgeretsi/smirk`, `bash quick_install.sh` for `SMIRK_em1.pt` + FLAME/eyelid/mediapipe assets).

As a library it mirrors `mica_local`:
```python
import smirk_local as smirk
h = smirk.load(device="cpu")                              # encoder + FLAME ONCE
smirk.compose(h, "photo.jpg", "identity.npy", "out.glb")  # MICA shape + SMIRK exp/pose -> .glb
```

## What's next

- Add the Stage-2 `transform(shape, key)` (a `local/face_hash.py` module) and wire it into `run.py`
  so `composed.glb` carries the *hashed* shape (today it's the raw MICA shape).
- Build the Stage-2 key-file layer on top (`enroll` averages `embed` over a person's photos →
  `arcface_centroid` + `source_shape`; see `../stage2-design.md` §3–§4).
