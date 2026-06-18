# ComfyUI_FaceHashDepth

Generates a **hash-clean, photo-aligned depth map on the pod** for the Stage-4 depth ControlNet —
the MICA-derived geometry that replaces the flat "neutral" scaffold. One photo IMAGE + a key in,
an aligned `depth` IMAGE + `mask` MASK out.

```
LoadImage -> LoadAndResizeImage -> FaceHashDepth(image, key, shape_source=hashed) -> depth ─┐
                                                                                            ▼
                                              InpaintCropImproved (same face mask) -> ControlNetApplyAdvanced
```

## Why it shells out (don't "fix" this)

MICA and SMIRK both ship top-level `utils/` `configs/` `datasets/` packages and **cannot be imported
into one interpreter** (see the repo's `CLAUDE.md` and `run.py`). So this node is an **orchestrator**:
it runs the two validated CLIs as subprocesses, exactly like `run.py`:

1. **MICA** (`hash_shape.py`): `photo + key` → hashed 300-d FLAME shape `.npy`
   (MICA encode → `arcface_keymix_v1(key)` on the embedding → decode).
2. **SMIRK** (`render_cond.py --maps`): `photo + shape` → `<stem>_depth.png` + `_mask.png`,
   projected with SMIRK's camera so it's **aligned to the photo**.

`shape_source`: `hashed` (per-key geometry), `own` (SMIRK's own recon — leaks original shape),
`neutral` (the generic mean head). Results are cached by `(image, key, offset, shape_source)`, so the
first render of a given key costs ~30–60 s and every run after is instant — including key sweeps.

## ⚠️ Same key as the InstantID node

Geometry (here) and texture (`FaceHashApplyInstantID`) are hashed with the **same** `arcface_keymix_v1`
but on **different** backbones (MICA's ArcFace vs antelopev2), so they aren't a co-designed single face
— but each is deterministic per key, so the combination is **consistent per key**. Set this node's
`key` (and `offset`) to the **same values as the InstantID FaceHash node**, or geometry and texture
drift apart.

## Pod setup

1. **Clone** into `ComfyUI/custom_nodes/`:
   ```bash
   cd /workspace/<...>/ComfyUI/custom_nodes
   git clone <this repo subdir>/ComfyUI_FaceHashDepth   # or symlink from the repo checkout
   ```
2. **The repo's `local/` must be present on the pod** with working MICA + SMIRK checkouts + weights,
   i.e. these must run by hand first:
   ```bash
   cd /workspace/<...>/face-hashing/local
   .venv/bin/python hash_shape.py in/<photo>.jpg --keys zack-secret -o /tmp/t       # MICA OK?
   .venv/bin/python render_cond.py /tmp/t/<photo>.png --maps --out-dir /tmp/t       # SMIRK OK?
   ```
   (FLAME 2020 weights are registration-gated/non-commercial — download per license, keep off git.)
3. **`cp config.example.json config.json`** and edit the paths:
   - `python_bin` — the interpreter that runs both CLIs (one env, subprocess isolation handles the collision)
   - `local_dir` — the repo's `local/` (holds `hash_shape.py`, `render_cond.py`, `mica_local.py`, `smirk_local.py`, `arcface_hash.py`, `neutral_shape.npy`)
   - `cache_dir`, `default_device` (`cuda` on the pod), `timeout`
4. **Restart ComfyUI.** The node appears under **FaceHash → FaceHash Depth (MICA→SMIRK)**.

## Wiring into M2_inpaint

Replace the depth `LoadAndResizeImage` (node 50) + the separate depth upload with this node fed by the
**same** `LoadAndResizeImage` that drives the rest of the graph; send its `depth` into the existing
parallel `InpaintCropImproved`, then `ControlNetApplyAdvanced`. Single photo input, no PNG shuffling.

## Gotchas

- **`device` on the pod.** The Mac path forces CPU ORT providers (`patch_mica_for_mac`); on CUDA you
  want the unpatched path. If `cuda` errors, try `cpu` (slow but correct) and fix the model env.
- **Filesystem-safe keys.** The cache name sanitizes the key, but `hash_shape.py` writes
  `photo__<key>.npy`, so avoid `/` in keys. The HASH always uses the raw key.
- **Cold start.** First render per `(photo,key)` loads SMIRK (+ MICA for `hashed`). Cached after.
- **No face / off-frame.** Raises with the subprocess stderr; check the input crop.

## Status

Phase-B scaffold. Core (`hash_shape.py` → MICA → distinct plausible head) is validated locally; this
packages it for the pod. Future: split SMIRK-encode caching (pose is key-independent) so key sweeps
skip re-encode; optional FaceHashEmbed unification if a shared backbone is ever adopted.
