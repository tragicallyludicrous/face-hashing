# Pod test — the single-input M2_inpaint workflow (FaceHashDepth)

How to bring up and verify the **single-photo-input** Stage-4 graph on RunPod for the first time.
`comfy/workflows/M2_inpaint.json` is already wired single-input: `LoadAndResizeImage(21)` fans into the
face-parse mask, InstantID, **and** `FaceHashDepth(54)`, whose depth runs through
`InpaintCropImproved(51) → ControlNetApplyAdvanced(53)`. Node 54 and the InstantID node (8) both ship
`key='zack-secret'`. So "testing" is really: get the `local/` stack onto the pod, prove the two CLIs run
there, install the node, run the graph.

> **Most of the risk is Phases 0–1, not ComfyUI.** Historically depth was pre-rendered on the Mac and
> uploaded as a PNG, so the pod has *never* had the MICA/SMIRK `local/` stack. Validate bottom-up — a
> failure in the two CLIs is far easier to debug at the shell than inside a node.

Paths below assume `REPO=/workspace/face-hashing` and `COMFY=/workspace/ComfyUI` — adjust to your pod.

---

## Phase 0 — Code + stack on the pod

```bash
cd /workspace
git clone git@github.com:tragicallyludicrous/face-hashing.git   # or: cd face-hashing && git pull
export REPO=/workspace/face-hashing
export COMFY=/workspace/ComfyUI                 # <-- your actual ComfyUI dir

cd $REPO
PYTHON=python3.11 ./setup.sh                    # venv, MICA+SMIRK, antelopev2, FLAME basis
```

`setup.sh` is **re-runnable** — every step checks before it fetches, so re-running on a persistent
`/workspace` volume just re-runs the doctor (it does **not** call SMIRK's unguarded `quick_install.sh`).
It runs `patch_mica_for_mac.py`, which forces **CPU** ONNX providers; on a CUDA pod that's fine —
first-run drives the depth node at `device=cpu` anyway (encode + numpy rasterizer ≈ 30–60 s on CPU; CUDA
is only a convenience here). Don't fight it yet.

Two gated weights `setup.sh` can't fetch (place once — it then shares FLAME across MICA+SMIRK and reuses
`mica.tar`, never re-downloading):
- FLAME `generic_model.pkl` → `local/MICA/data/FLAME2020/`
- MICA `mica.tar` → `local/MICA/data/pretrained/`

### Snags a fresh pod actually hits (now handled by the script)
- **No `python3.11`.** Minimal/PyTorch RunPod images ship 3.10. Install once, then re-run setup:
  ```bash
  apt-get update && apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev
  PYTHON=python3.11 ./setup.sh                   # or point PYTHON at any 3.11 binary
  ```
- **No `unzip` / `wget`.** Needed to extract a manually-downloaded `FLAME2020.zip`; `setup.sh` step 0
  apt-installs them on the pod.
- **`gdown` missing or `--id` error.** `gdown` is a *venv* package (the doctor's package probe doesn't
  list it) and newer gdown dropped `--id`. `setup.sh` self-heals the install and uses the URL form. The
  manual fetch, if you need it:
  ```bash
  local/.venv/bin/python -m gdown "https://drive.google.com/uc?id=<id>" -O <path>
  ```
- **mediapipe `libGLESv2.so.2: cannot open shared object file`.** Minimal pods lack the GL runtime libs
  mediapipe's vision tasks `dlopen` (even for CPU). `setup.sh` step 0 apt-installs
  `libgl1 libglib2.0-0 libgles2 libegl1`.
- **Slow `import torch` on a network-volume pod (~20 s+).** `/workspace` is MooseFS; importing the 2 GB
  CUDA torch is thousands of remote metadata RPCs. Build the venv on **local disk** reusing ComfyUI's
  torch: `FACEHASH_VENV=/opt/face-venv FACEHASH_BASE_PY=/usr/bin/python3.12 ./setup.sh`, then point the
  node's `config.json` `python_bin` (and `FACEHASH_PY`) at `/opt/face-venv/bin/python`.

```bash
python3 viewer/studio_server.py --check         # all [OK] = Phase 0 done, go to Phase 1
```

## Phase 1 — Prove the two CLIs run on the pod (the real test) — ✅ validated on the pod

This is exactly what `FaceHashDepth` shells out to. If these produce a depth PNG by hand, the node works.
Use the venv `setup.sh` built (`PY` below = `/opt/face-venv/bin/python` in reuse mode, else `.venv/bin/python`).

```bash
cd $REPO/local
PY=/opt/face-venv/bin/python                     # reuse-mode venv; else local/.venv/bin/python
PHOTO=in/Zack-from-below.jpeg                    # any face photo present on the pod
STEM=$(basename "${PHOTO%.*}")

# MICA: photo + key -> hashed 300-d shape
$PY hash_shape.py "$PHOTO" --keys zack-secret --offset 0.0 -o /tmp/fh --device cpu
ls /tmp/fh/                                      # expect  ${STEM}__zack-secret.npy

# SMIRK: photo + that shape -> aligned depth/mask.
# Pass the ONE shape file for THIS photo — a glob (*__zack-secret.npy) matches leftover .npy
# from earlier runs and argparse rejects the extras.
$PY render_cond.py "$PHOTO" --maps --out-dir /tmp/fh \
    --shape "/tmp/fh/${STEM}__zack-secret.npy" --device cpu
ls /tmp/fh/${STEM}_depth.png /tmp/fh/${STEM}_mask.png   # expect both
```

Open `${STEM}_depth.png`: a face-shaped depth aligned to the photo's pose. **If this fails, stop here** —
fix the weight / no-face / detector issue before touching ComfyUI. (Benign noise to ignore: HF
"unauthenticated requests" notices, timm "Unexpected keys … classifier/conv_head", and FLAME/numpy
`VisibleDeprecationWarning: align=0` from unpickling the FLAME model.)

## Phase 2 — Install the node into ComfyUI

```bash
# Symlink the repo copy so git pulls keep it in sync (don't copy)
ln -s $REPO/comfy/custom_nodes/ComfyUI_FaceHashDepth \
      $COMFY/custom_nodes/ComfyUI_FaceHashDepth

cd $COMFY/custom_nodes/ComfyUI_FaceHashDepth
cp config.example.json config.json
```

Edit `config.json` to match the pod and set device to **cpu** for the first run:

```json
{
  "python_bin":    "/workspace/face-hashing/local/.venv/bin/python",
  "local_dir":     "/workspace/face-hashing/local",
  "neutral_shape": "/workspace/face-hashing/local/neutral_shape.npy",
  "cache_dir":     "/workspace/face-hashing/local/.facehashdepth_cache",
  "default_device":"cpu",
  "timeout": 900
}
```

**Restart ComfyUI.** Watch the startup log for `ComfyUI_FaceHashDepth` import errors. The node appears
under **FaceHash → "FaceHash Depth (MICA→SMIRK)"**.

## Phase 3 — Smoke-test the node alone

Throwaway 3-node graph: `LoadImage → FaceHashDepth → PreviewImage` (+ a MaskPreview on the mask output).
`shape_source=hashed`, `key=zack-secret`, `device=cpu`. Queue it.

- First run ≈ 30–60 s (cold MICA+SMIRK load); cached thereafter.
- Confirm a clean, face-aligned depth preview. This isolates the node from the rest of M2_inpaint.

## Phase 4 — Run the full M2_inpaint graph

1. **Load it:** drag `$REPO/comfy/workflows/M2_inpaint.json` onto the canvas. The sidebar won't list it —
   it lives in the repo's `comfy/workflows/`, not ComfyUI's `user/default/workflows/` (the only dir the
   sidebar scans).
2. **Same-key rule:** node 54 (FaceHashDepth) and node 8 (FaceHashApplyInstantID) must share `key` **and**
   `offset` — both ship `zack-secret` / `0.0`. Geometry (MICA) and texture (antelopev2) are hashed on
   different backbones; they only stay a consistent per-key pair if the key/offset match.
3. Set node 54's `device` to **cpu** for this run.
4. Confirm the rest of the existing pod setup: node 8 `embedding_path`
   (`/workspace/models/identities/zack.npy`), the two ControlNets
   (`controlnet-depth-sdxl-1.0.safetensors` on 52, `instantid-controlnet-sdxl.safetensors` on 7), and that
   `LoadAndResizeImage(21)` points at a photo in `ComfyUI/input/`.
5. **Queue Prompt.** Watch the console — the node streams subprocess stderr on failure.

## Phase 5 — Verify the result *and* the hash property

- Face lands in the right place, depth-guided — no whole-head-in-the-circle, no over-blend (the failure
  this depth bridge exists to fix).
- **Determinism:** change the key on **both** nodes (e.g. `zack-test2`) → a different but coherent face
  (≈ 30–60 s cold, cached after). Switch back to `zack-secret` → the *same* face returns. That round-trip
  is the point of the stage.
- After editing geometry, flip node 54 `force_rerender` true once (cache key =
  image+key+offset+shape_source+device).

## Phase 6 — On-manifold synthetic identities (`arcface_keymix_whitened_v3`, recommended)

The default `arcface_keymix_v1` is a signed permutation → **off-manifold** → InstantID renders inhuman
faces on some inputs (and forces you to back `ip_weight`/`cn_strength` down). `arcface_keymix_whitened_v3`
keymixes in a **whitened identity subspace** (where a permutation preserves the distribution), so the
output is an on-manifold **derived synthetic** identity — no real face targeted. One-time setup:

```bash
cd /workspace/face-hashing/local
PY=/opt/face-venv/bin/python

# 1) download synthetic faces — SFHQ (fully synthetic, no real identity). ~1500 faces, margin-padded
#    so RetinaFace detects them (SFHQ are tight crops -> padding lifts detection ~30% -> ~100%).
#    Needs only huggingface_hub + pillow (no `datasets`).
$PY fetch_synthetic_faces.py -o synthetic_faces/        # bigger corpus: --zips SFHQ-part1.zip -n 5000

# 2) extract embeddings (antelope + mica), then fit the whitening basis (pure numpy, fast)
$PY build_gallery.py -i synthetic_faces/ -o corpus.npz --device cpu
$PY build_basis.py  -i corpus.npz -o basis.npz --var 0.95     # -> basis.npz (aggregate stats; commit-safe)
```

Sanity-check the transform before ComfyUI (a real face npy → a different, in-distribution synthetic):
```bash
$PY arcface_hash.py <some>_arcface.npy --transform arcface_keymix_whitened_v3 \
    --basis basis.npz --column antelope -k zack-secret -o /tmp/v3.npy   # prints cos(original,hashed)
```

**Wire both nodes (must match):** on `FaceHashApplyInstantID#8` **and** `FaceHashDepth#54` set
`transform = arcface_keymix_whitened_v3`, the same `key`, the same `basis_path` (point at `basis.npz`),
and the same `offset`. Each node auto-reads its column — InstantID `antelope`, depth `mica` — so you only
set one path. (Your DRY key primitive already shares the key; add a String primitive for `basis_path`.)

**Bring-up order:** Phase-3 isolation first — `LoadImage → FaceHashDepth(v3, basis_path set) →
PreviewImage`. Confirm a clean depth, *then* run the full graph. With on-manifold embeddings, push
`ip_weight`/`cn_strength` back up toward ~0.8; the OOD monsters should be gone.

**Caveat:** v3 co-design is approximate — MICA and antelopev2 whiten in different spaces, so the same key
gives a plausible synthetic in *each* but not provably the *same* person. Per-key consistency holds within
each path; a learned cross-backbone map would tie them (future).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Node missing after restart | import error in console — usually `local_dir` wrong in `config.json`, or a node-env dep missing (`mediapipe`, `scikit-image`) |
| "subprocess failed … no face" | Phase-1 CLIs not actually green, or the photo has no detectable face |
| CUDA / ORT provider error on node 54 | you set `device=cuda`; `patch_mica_for_mac` forces CPU ORT — use `device=cpu`, or remove the patch and fix the model env |
| Depth misaligned with the face crop | node 54 must be fed the **same** `LoadAndResizeImage(21)` output as the rest of the graph (it is, in the repo file) |
| Geometry ≠ texture identity | node 54 and node 8 `transform`/`key`/`offset`/`basis_path` diverged (must match) |
| First render slow every time | cache not persisting — check `cache_dir` is writable and stable across runs |
| `needs basis_path` / `basis … missing` | `transform=arcface_keymix_whitened_v3` but no/!valid `basis.npz` — run Phase 6 build; the `.npz` needs `antelope_*` (+ `mica_*` for depth) keys |
| Still inhuman with `v3` | basis fit on too few / non-face images — use more synthetic faces; or `offset` too high (start 0); confirm `arcface_hash.py … --transform arcface_keymix_whitened_v3` sanity-check looked plausible |

## Reference

- Node internals + same-key rationale: `comfy/custom_nodes/ComfyUI_FaceHashDepth/README.md`
- Design/scope (phases, MICA consistency mechanism, gotchas): `stage4-pod-depth-bridge.md`
- CLIs the node orchestrates: `local/hash_shape.py` (MICA), `local/render_cond.py` (SMIRK)
