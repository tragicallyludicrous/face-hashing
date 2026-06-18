# Stage 4 — Pod-side Hashed-Depth Node (scope)

*Written 2026-06-18. Goal: one photo input on the ComfyUI graph drives mask + hashed identity
+ aligned depth, all on the pod — and (Phase B) keyed, so the depth geometry tracks the same key
as the InstantID identity. Removes the separate depth-PNG upload and the dual-`LoadAndResizeImage`
alignment hack. Read `CONTEXT.md` + `stage4-quality.md` first.*

---

## 0. Where we are

- **`local/render_cond.py` (Mac):** SMIRK encode → orthographic-project with SMIRK's cam + the
  face-crop transform → **pure-numpy z-buffer rasterize** → `_depth.png` / `_mask.png` / `_normal.png`,
  aligned to the photo. No PyTorch3D. `--shape` overrides geometry (neutral / SMIRK's own / hashed).
- **ComfyUI (pod):** `FaceHashApplyInstantID` already hashes the ArcFace embedding **on the pod**.
  But the **depth** is pre-rendered on the Mac and uploaded as a PNG; node 50 (a 2nd
  `LoadAndResizeImage`) + node 51 (a parallel `InpaintCropImproved`) re-align it.
- **The gap:** depth generation isn't on the pod, so (a) you shuffle PNGs, and (b) the key cannot
  drive the geometry — the depth is a fixed scaffold, independent of the FaceHash key.

The neutral scaffold (`render_cond --shape <zeros-300>`) is the hash-clean *fixed* version:
generic head, photo's pose/expression, no original-identity geometry, key-independent. Good enough
to iterate keys today; this doc is about folding depth generation **into the graph**.

---

## 1. Target graph — single photo input

```
LoadImage(photo) ─► LoadAndResizeImage(working res) ─┬─► Face-Parsing subgraph ─► MASK
                                                     ├─► ID Transform subgraph (FaceHash→InstantID)
                                                     └─► Depth Scaffold subgraph (NEW) ─► depth IMAGE + MASK
                                                                                              │
                                                            ControlNet subgraph  ◄────────────┘
```

One input, three consumers. **Node 50 and the dual-load alignment hack disappear** — the depth is
born in-graph at working resolution, so the existing parallel `InpaintCropImproved` aligns it exactly.
Matches the subgraph grouping already in the workflow (Face-Parsing / ID Transform / ControlNet) by
adding a peer **"Depth Scaffold"** subgraph.

---

## 2. The node(s), phased

### Phase A — single photo input, geometry NOT keyed (ships the single-input win)
`FlameDepthScaffold`:
- **inputs:** `image` (the working IMAGE), `shape_source ∈ {neutral, smirk_own}`, `device`
- **does:** SMIRK encode (pose/exp/cam + crop tform) on the image → pick shape (zeros = neutral, or
  SMIRK's own reconstruction) → project + rasterize (ported from `render_cond`) 
- **outputs:** `depth` IMAGE, `mask` MASK (, `normal` IMAGE)

This alone delivers "run from a single photo input." Geometry is **hash-clean (neutral)** or **own**.
No MICA, no embedding plumbing. Lowest-risk first build.

### Phase B — hashed geometry, fully keyed
Add `MicaShapeFromEmbedding`: ArcFace embedding (512-d) → 300-d FLAME shape (MICA encode). Then
compute the hash **once** and fan it out:

```
photo ─► FaceHashEmbed(key, offset) ─► hashed 512-d ─┬─► InstantID            (identity / texture)
                                                     └─► MICA ─► 300-d shape ─► FlameDepthScaffold(shape override) ─► depth (geometry)
```

`shape_source` gains a `hashed` option. One key now drives texture **and** geometry.

---

## 3. Why MICA (the consistency mechanism)

ArcFace(512-d) and FLAME-shape(300-d) are **different spaces**; hashing each independently yields two
unrelated faces (head shape says person X, texture says person Y). **MICA is the deterministic
ArcFace→shape map** — so deriving the shape *from the hashed embedding* makes geometry and texture the
**same** synthetic identity. Rule: don't hash the shape directly, **derive** it from the hashed embedding.

---

## 4. Dependencies / weights on the pod

- **SMIRK** checkout + weights (encoder + generic FLAME). FLAME 2020 is **registration-gated,
  non-commercial** — keep weights **off git**, download per license (same constraint as local).
- **MICA** checkout + weights (Phase B; research-only).
- `torch` (present), `mediapipe`, `scikit-image`, `opencv` — add to the node package requirements.
- Port `project_to_image`, `rasterize`, `_vertex_normals` from `render_cond.py` — pure numpy, trivial.

---

## 5. Gotchas

- **No global `chdir`.** `smirk_local` chdir's into its checkout on import (fine for a one-shot CLI;
  **breaks a long-running ComfyUI process** and every other node). The port must load SMIRK/FLAME with
  **absolute paths** and never chdir.
- **Detector parity.** Keep the mediapipe → RetinaFace fallback (small/profile faces mediapipe misses)
  so the in-graph encode matches `render_cond`.
- **Resolution.** Feed the node the `LoadAndResizeImage` **output** (working res), so depth is born at
  working res — the parallel `InpaintCropImproved` aligns it exactly; delete node 50 + the dual-load.
- **Mouth cavity = black (−inf depth).** Expected (open-mouth interior); the SDXL depth CN handles it.
- **CUDA.** SMIRK/MICA **encode** paths are PyTorch3D-free and the rasterizer is numpy — CUDA on the
  pod is a *convenience* (faster encode), **not** a requirement. The "CUDA blocker" never touched render.

---

## 6. Packaging + subgraph layout

New custom-node package **`ComfyUI_FaceHashDepth`** (sibling to `ComfyUI_FaceHash`), vendoring the
SMIRK/MICA encode wrappers + the rasterizer. Add a **"Depth Scaffold"** subgraph to match the existing
grouping; the single `LoadImage`→`LoadAndResizeImage` fans into it.

---

## 7. Build checklist

- [ ] Port rasterizer (`render_cond` → node module), **no chdir**, absolute paths
- [ ] `SmirkEncode` wrapper → pose/exp/cam + crop tform from the working IMAGE
- [ ] `FlameDepthScaffold` node (`shape_source` neutral|own) → depth/mask  **← Phase A ships here**
- [ ] Rewire: single `LoadImage` → Depth Scaffold subgraph; **delete node 50 + dual-load**
- [ ] `MicaShapeFromEmbedding` + `FaceHashEmbed` fanout  **← Phase B**
- [ ] `shape_source=hashed`; validate geometry+texture depict **one** identity across keys

---

## 8. Open questions

- **Embedding exposure:** add an EMBEDDING output to the existing `FaceHashApplyInstantID`, or split a
  `FaceHashEmbed` node and rewire InstantID to consume a precomputed embedding? (Phase B refactor.)
- **Caching:** cache SMIRK encode per photo so key-only sweeps don't re-encode (only re-hash → re-MICA →
  re-raster). Matters for fast key A/B.
- **Default scaffold:** neutral (hash-clean, generic proportions) vs SMIRK's own (better fit, partial
  shape leak) as the Phase-A default.
