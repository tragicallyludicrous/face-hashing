# Stage 2 design — the identity key file & the deterministic transform

Status: **design, not yet implemented.** This is the authoritative plan for Stage 2 (the "hash"
itself) and the deployment-topology shift that comes with it. Supersedes the Stage-2 sketch notes
in `CONTEXT.md` / `face_hashing_research_report.md` where they conflict. Stage 1 (photo → MICA/DECA
identity vector) is the input to everything below.

---

## 1. The problem this solves

The hash promise is **determinism**: the same person's face must always map to the same output
face. Measured reality (see `consistency_report`): identity vectors *cluster* per person but never
*coincide* — MICA identity scored AUC ≈ 0.94, ArcFace ≈ 1.00, on a small set. So a naive
`transform(shape, key) → shape'` run on each photo would hash a person's several photos to several
*different* outputs, because the input `shape` wobbles photo-to-photo. The wobble is the enemy.

Two ways to absorb the wobble:

- **Quantize (stateless):** snap the vector to a grid/codebook so a cluster collapses to one bucket.
  Keeps a pure function, but bucket *boundaries* are fragile (two photos straddle a boundary → two
  outputs) and two strangers can share a bucket.
- **Enroll + look up (stateful):** decide identity once, store it, retrieve it forever after. The
  database version of this puts the index on a server (1:N search, collisions grow with N, and *you*
  hold everyone's biometrics). **This document chooses a third form: a per-user, encrypted key file
  — a decentralized index of size one.**

The key file is just an adaptive quantizer whose single bucket is the user's enrolled centroid,
carried by the user instead of stored centrally. It turns **1:N identification into 1:1
verification**, which is a far easier, collision-free, trivially-scalable problem.

---

## 2. Two modes

The presence or absence of a key file selects the behavior — same pipeline, two entry points.

| | **Mode A — recurring user** | **Mode B — one-off / for fun** |
|---|---|---|
| Key file | present (user supplies it + passphrase) | none |
| Identity source | **read from the file** (`hashed_shape`) | computed live: `transform(MICA_shape, default_key)` |
| Cross-photo determinism | **perfect** (identity never recomputed) | best-effort (subject to per-photo wobble) |
| Who can run it | only the enrolled person (ArcFace-gated) | anyone, any photo |
| State stored | the user's own file; server stores nothing | nothing |
| Use case | "generate new content as *my* alt-identity" | "swap a stranger's face once" |

**Mode A flow**
1. Decrypt the key file with the passphrase.
2. Run ArcFace on the incoming photo; **verify** `cos(embedding, arcface_centroid) ≥ match_threshold`
   (1:1). Fail → "couldn't confirm this is you" (re-enroll or fall back to Mode B).
3. Read `hashed_shape` **straight from the file** — *do not recompute it from the photo.* This is
   what makes Mode A perfectly deterministic: the wobbly path (photo → MICA) never touches identity.
4. Take pose / expression / lighting from the *current* photo (DECA `exp`/`pose`, or let Stage-4
   diffusion condition on the photo directly).
5. Composite (Stage 3 + 4): decode `hashed_shape` into the photo's pose → photorealize → paste back.

**Mode B flow**
1. MICA (or DECA) on the photo → `shape`.
2. `transform(shape, default_key) → hashed_shape`.
3. Composite. Nothing stored. Optionally offer "save as a key file" → single-photo enrollment
   (lower quality than multi-angle, but bootstraps a Mode-A credential).

---

## 3. The identity key file

A standalone, portable, **passphrase-encrypted** credential — the SSH-private-key analogy, made
precise. "Encrypted like an SSH key" means *symmetric encryption of the contents, unlocked by a
passphrase* (à la `ssh-keygen -p`), **not** an asymmetric keypair. What it protects is the
**biometric linkage** between the real person and their alt-face.

### 3.1 Plaintext payload (before encryption)

```jsonc
{
  "version": 1,
  "created": "2026-06-01T...",
  "arcface_centroid": [ ...512 floats... ],   // mean of enrollment embeddings, renormalized; 1:1 verify
  "match_threshold": 0.35,                     // cosine cutoff for the verify gate (tunable)
  "extractor": "mica",                         // which identity model produced source_shape
  "source_shape":  [ ...300 floats... ],       // averaged enrolled identity (provenance / re-deriveable)
  "hashed_shape":  [ ...300 floats... ],       // THE deterministic alt-identity — the payload
  "transform": "flame_shape_offset_v1",        // which hash strategy produced hashed_shape
  "key_seed": "base64..."                      // secret seed the transform consumed (optional; for regen)
}
```

- `arcface_centroid` + `match_threshold` → the **verify gate** (who may use this file).
- `hashed_shape` → the **payload** (the sealed alt-identity; Mode A reads this and stops).
- `source_shape` + `transform` + `key_seed` → **provenance / regeneratability** (recompute
  `hashed_shape` if you swap transforms, or audit how it was made).

### 3.2 Encryption

Boring, vetted, off-the-shelf — **do not roll your own.**
- KDF: **argon2id** (passphrase → 256-bit key).
- AEAD: **AES-256-GCM** (or `cryptography`'s `Fernet`, or wrap the whole file with `age`).
- File = `salt ‖ nonce ‖ ciphertext ‖ tag`. A stolen file is inert without the passphrase.

The passphrase is the real access control. The ArcFace gate is a *secondary* binding: even a
decrypted file only works on the rightful face (you can't point Alice's key at Bob). It is **not**
strong security — "passphrase + any photo of Alice that clears the threshold" suffices. Fine for a
creative tool; don't oversell it.

### 3.3 Multiple personas

Like holding several SSH keys: one person can keep several key files, each with a different
`key_seed`/`transform` → a different consistent alt-identity ("public persona," "anon persona").
Free, because the transform is keyed.

---

## 4. Enrollment (builds a key file)

Cooperative, FaceID-style — the one step that needs the subject's participation. Reuses pieces that
already exist in `pipeline.py` (`arcface_embed`, the MICA extractor, the FLAME decoder).

1. **Capture** a few photos at different angles (3–6).
2. **Confirm same person:** embed each with `arcface_embed`; check they mutually pass
   `match_threshold` (reject outliers — a stray photo of someone else). This is the enrollment-time
   analog of `consistency_report`'s intra-person check.
3. **ArcFace centroid:** mean of the embeddings, **L2-renormalized** → `arcface_centroid`.
4. **MICA shape average:** run MICA on each photo, **average in shape-coefficient space**
   (FLAME shape is linear/PCA, so the mean of identity vectors is itself a valid face — *do not*
   average posed meshes vertex-wise, that blurs) → `source_shape`.
5. **Hash once:** `hashed_shape = transform(source_shape, key_seed)`.
6. **Seal:** serialize the payload, encrypt with the passphrase, write `<name>.fhash`.

Multi-view averaging also yields a *better* mesh than any single photo (it resolves single-view
depth ambiguity) — a quality win independent of determinism.

---

## 5. The transform (the "hash function") and where it slots in

`pipeline.default_mutation(shape)` is the current placeholder (it flips/exaggerates a couple of PCs).
Stage 2 generalizes it to a **keyed, hot-swappable registry**:

```python
def transform(shape, key, strategy="flame_shape_offset_v1"):
    return _TRANSFORMS[strategy](shape, key)   # -> shape' (same dtype/shape)
```

- **v1 — seeded Gaussian offset on identity.** `key` seeds an RNG → a fixed offset vector in shape
  space; add to `shape`, clamp each coeff to ±2–3σ so you stay on the face manifold. Deterministic
  in `(shape, key)`. Mutate identity only; never touch `exp`/`pose`.
- Later strategies (rotation in a subspace, codebook projection, etc.) register under new names.

**Framing discipline (keep `CLAUDE.md` honest):** the *transform* is **obfuscation, not
encryption** — it does not become cryptographic just because the *key file* around it is encrypted.
Those are two separate layers: the file gets real crypto (§3.2); the face-mutation stays a keyed
geometric move. Don't conflate them.

### Planned `pipeline.py` surface (additions)

```
# identity extraction (Stage 1) — exists / to add
arcface_embed(image_path)                       # exists
mica_embed(image_path)                          # to add: MICA shape, no renderer (mirrors load_deca)

# the hash (Stage 2)
transform(shape, key, strategy=...)             # registry; default_mutation folds in as a strategy
enroll(image_paths, passphrase, out_path, ...)  # §4 -> writes <name>.fhash
load_identity(path, passphrase)                 # decrypt + verify schema -> payload dict
verify(payload, image_path)                     # 1:1 ArcFace gate -> bool/score

# orchestration (the two modes)
hash_face(image_path, key=None, passphrase=None)  # key present -> Mode A, else Mode B
```

---

## 6. Determinism guarantees & failure modes

- **Mode A: exact.** Identity is decrypted, not recomputed; two photos of the same person yield
  byte-identical `hashed_shape`. The only soft edge is the verify gate occasionally false-rejecting a
  genuine photo (tune `match_threshold`; fall back to using it anyway or re-enroll).
- **Mode B: best-effort.** No enrollment → per-photo wobble leaks into identity. Acceptable for a
  one-off, by definition.
- **Lost key = lost identity.** The classic "lost my SSH key" problem, and there's no clean recovery:
  re-enrolling produces a *different* alt-face (we can't reproducibly re-derive the seed from the
  person — that's the very problem we couldn't solve). Accept this consciously.
- **Threshold tuning.** Too tight → same person fails their own gate (non-determinism returns); too
  loose → a look-alike passes. 1:1 makes this forgiving, but identical twins remain the known failure.

---

## 7. Deployment topology — local-first, with one remote GPU call

The "everything in Colab" setup was never load-bearing; it was a **workaround for the PyTorch3D /
CUDA-rasterizer build**, which we already eliminated by **not rendering** (`load_deca` neuters
`_setup_renderer`). Drop rendering and most of the pipeline stops needing CUDA at all:

| Stage | Needs a GPU? | Where it can run |
|---|---|---|
| 1. Identity extract (MICA/DECA encode + FLAME decode) | **No** (ResNet + linear FLAME; no render) | **Mac, native** — CPU or Apple **MPS** |
| ArcFace embed (insightface / onnxruntime) | No | **Mac** — already runs on CPU; CoreML provider available |
| 2. Transform (numpy/torch vector math) | No | **Mac**, trivial |
| 3. FLAME decode → mesh (`_decode_verts`) | No | **Mac** |
| Enrollment + key-file crypto | No | **Mac** |
| Viewer | No | **Mac** (already) |
| **4. Diffusion photorealism (InstantID / PuLID / Arc2Face)** | **Yes — genuinely** (SDXL-class) | **Remote GPU API** |

So the real answer to "call an API for the CUDA steps" is: **there's essentially one CUDA step —
Stage 4** — and that's the one to put behind an API. Everything else becomes a local Python program
on the Mac.

### 7.1 Porting Stages 1–3 to the Mac

- **The main task is the same surgery we did for DECA:** MICA imports PyTorch3D at module load for
  *rendering only*; stub/lazy-import it so the identity path never needs it (mirror
  `DECA._setup_renderer = no-op`). Then MICA is just torch + insightface + trimesh + the chumpy shim
  + the FLAME `.pkl` — all of which run on macOS.
- **Device:** CPU works for a handful of images (enrollment is once-per-user; Mode B is one image).
  Apple **MPS** speeds it up; set `PYTORCH_ENABLE_MPS_FALLBACK=1` since a few ops still fall back to
  CPU. Unverified end-to-end on Mac yet — **treat as "should work, must test."**
- The Drive-cache / Colab gotchas in `CONTEXT.md` become irrelevant locally (no ephemeral runtime);
  keep them only as long as Colab stays a parallel path.

### 7.2 The remote diffusion call

Options, easiest → most control:
- **Hosted endpoints (Replicate, fal.ai):** ready InstantID/PuLID/Arc2Face models, pay-per-call, no
  infra. Best for getting Stage 4 working fast.
- **Serverless GPU you control (Modal, RunPod):** wrap your own pipeline in a function on an
  A10G/L4; call over HTTP. Choose when you need a custom Arc2Face/compositing graph.
- Self-hosted VM with a GPU: most control, most babysitting; skip unless needed.

### 7.3 Privacy decomposition (matters *specifically* for this project)

This tool *de-identifies* faces, so shipping the original face to a third-party GPU is
self-defeating. Because identity extraction and the transform run **locally**, the cloud never needs
to see who the person really is — it only needs the **new (hashed) identity** to render. The
privacy-preserving split:

- **Cloud receives:** the hashed identity conditioning (decoded mesh / ArcFace embedding of
  `hashed_shape`) + a *pose/landmark target or mask* — i.e., the synthetic replacement and where it
  goes, **not** the real face.
- **Mac keeps & does locally:** the original photo, identity extraction, the transform, and the
  **final composite** (paste the generated face crop back into the original).

Net: the remote GPU only ever sees a face that doesn't belong to anyone real. Prefer a self-hosted
endpoint (Modal/RunPod) over a hosted one if even the synthetic-face round-trip feels like too much
exposure.

---

## 8. Open questions / next steps

1. ~~**Verify MICA runs on the Mac**~~ — **DONE (2026-06-01).** Stage 1 runs natively on Apple
   Silicon; CPU output matches Colab at **cosine 1.0** on the same photo. No renderer/PyTorch3D
   needed — the inference path is clean. Runner + runbook in `local/` (`patch_mica_for_mac.py`,
   `README.md`); MPS works too (needs a float32 input cast, handled by the patcher). The local-first
   pivot is green.
2. **Pick `match_threshold`** empirically from `consistency_report` data (ArcFace intra vs inter on a
   larger, matched set) rather than guessing 0.35.
3. **Implement `transform` v1** (seeded Gaussian offset on `shape`, ±2–3σ clamp) + the registry; fold
   `default_mutation` in as a strategy.
4. **Implement the key-file layer** (`enroll` / `load_identity` / `verify`) with argon2id + AES-GCM.
5. **Stand up Stage 4** behind one API call (start with a hosted InstantID/PuLID endpoint), then wire
   the privacy decomposition in §7.3.
6. Decide whether Colab stays a parallel path or is retired once Stages 1–3 run natively on the Mac.
