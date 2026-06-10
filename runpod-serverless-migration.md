# RunPod Serverless Migration Manual

How to move Face-Hashing's Stage-4 inference off SwarmUI and onto a **reproducible
RunPod serverless ComfyUI endpoint**, design the graphs at home, and sync painlessly.

Written after two template breakages (huggingface_hub 1.0; the mmap `get_file_handle`
crash) — the whole point of this setup is that those can't happen again, because the
worker is built from a **pinned Dockerfile**, not mutated live.

---

## Which loop am I in? (read this BEFORE anything else)

There are two completely different loops here. Using the wrong one is the single biggest
time-sink, so decide first:

- **🛠 BUILDING the workflow** (adding/tweaking nodes, downloading & testing models, tuning the
  graph — *almost all of the work*): use a **persistent GPU pod + the network volume** (§6). Live
  ComfyUI canvas; install nodes via Manager/git, pull models straight onto the volume, run, repeat.
  **No Docker. No image rebuilds. Nothing multi-GB moves per change.** Stop the pod when idle; the
  volume keeps everything.
- **📦 SERVING a finished workflow** (cheap scale-to-zero inference, an always-on API): bake the
  *stabilized* node set into the **serverless image** (§3–5). You build/push that image **rarely** —
  only when the node/dep set changes, which by then is seldom.

> Rule: **never iterate through the serverless Docker image.** The build→push (≈9 GB)→redeploy
> cycle is a *packaging* step for a finished workflow, not a development loop. If you're still
> changing nodes, you're in the pod loop. Sections 3–5 are a destination, not a daily routine.

The **network volume** is the through-line: the models you download while iterating on the pod are
already there when you eventually build the serverless image, so there's no migration — only the
(by-then-stable) custom nodes + deps go into the image.

---

## 0. The architecture (read this first)

```
  MAC (home)                          GIT (GitHub)                 RUNPOD
  ──────────                          ────────────                 ──────
  build_m2.py  ──┐                                          ┌─ Network Volume ─┐
  build_*.py     ├─ to_comfy_api.py ─► comfy/api/*.json ──► │  models/ (HF)    │
  (canonical)    │  to_comfy.py     ─► comfy/comfyui/*.json │  (shared)        │
                 │                                          └────────┬─────────┘
  ComfyUI_FaceHash ─────────────── git push ───► git pull ──────────┤
  (custom node)                                                     │
                                                       ┌────────────┴───────────┐
  local photos ── rclone/Dropbox ──────────────►       │  Serverless Endpoint   │  ◄─ /runsync  (inference)
  outputs      ◄─ rclone/Dropbox ──────────────        │  (pinned worker image) │
                                                       │  Dev Pod (occasional)  │  ◄─ interactive ComfyUI
                                                       └────────────────────────┘
```

Roles:

- **Mac** — design + canonical source. The repo already lives in Dropbox, so it's backed up at home automatically. You author graphs as **code** (`build_*.py`), not by hand.
- **Git** — the transport for *workflows and node code* to RunPod. Small text, versioned, robust.
- **Network volume** — the shared **model store** (and, if you like, a dev ComfyUI). Mounted `/workspace` on a pod, `/runpod-volume` on serverless.
- **Serverless endpoint** — a pinned worker image that runs ComfyUI headless and answers `/runsync`. This is production inference.
- **Dev pod (occasional)** — a cheap GPU pod on the *same* image + volume for interactive canvas work when you need to iterate fast. Stopped when idle.
- **rclone/Dropbox** — for the *private binaries* (input photos, generated outputs) that don't belong in git.

Three generators, one source of truth (`build_*.py`):

| Script | Output | Used by |
|---|---|---|
| `to_comfy.py` | `comfy/comfyui/*.json` (litegraph UI graph) | open in a ComfyUI **canvas** |
| `to_comfy_api.py` | `comfy/api/*.json` (flat API prompt) | **POST to serverless** |
| `to_swarm.py` | SwarmUI envelope | *retired* (see §1) |

---

## 1. Migrate off SwarmUI → plain ComfyUI (do this first; it's mostly deletion)

SwarmUI was only ever the *frontend*. Nothing about the hash, the FaceHash node, or the
models depends on it. Because your workflows are **code-generated**, "migrating" is mostly
standardizing on the ComfyUI-native generators and dropping the envelope.

1. **Stop generating SwarmUI envelopes.** You no longer run `to_swarm.py`. Keep the file
   in the repo for reference, or delete it — your call. The canonical artifacts are now
   `comfy/comfyui/*.json` (UI) and `comfy/api/*.json` (serverless).

2. **(Optional) Install a plain ComfyUI on the Mac for design.** You only need this if you
   want an interactive canvas at home; otherwise design on the dev pod (§6).
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
   cd ~/ComfyUI
   python3.11 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt          # torch picks MPS automatically on Apple Silicon
   python main.py                            # http://127.0.0.1:8188
   ```
   Then drop the FaceHash node in and open a workflow:
   ```bash
   cp -R "<repo>/comfy/custom_nodes/ComfyUI_FaceHash" ~/ComfyUI/custom_nodes/
   git clone https://github.com/cubiq/ComfyUI_InstantID ~/ComfyUI/custom_nodes/ComfyUI_InstantID
   # File ▸ Open ▸ <repo>/comfy/comfyui/M2_portrait_fp16.json
   ```
   On the Mac this is for *graph design*, not full inference — SDXL+InstantID on MPS is slow.

3. **Retire SwarmUI's CustomWorkflows.** Your workflows live in the repo now, not in
   `SwarmUI/src/.../CustomWorkflows/`. You can leave the SwarmUI install parked or remove it
   once you're comfortable; there's no data in it you don't already generate from `build_*.py`.

> Why this is safe: the only thing SwarmUI gave you was auto-generated parameter widgets.
> You drive parameters from `build_*.py` instead, which is more reproducible anyway.

---

## 2. Build the network volume (shared model store)

1. **Create a network volume** in the RunPod **region** you'll deploy in (serverless and the
   volume must match region). Size it for your model set — 50–100 GB is plenty for
   SDXL + InstantID + a few checkpoints.

2. **Spin up a cheap GPU pod** with that volume attached (it mounts at `/workspace`). Use any
   ComfyUI or CUDA pod template; this pod is just for populating the volume + dev (§6).

3. **Pull models onto the volume** (zero egress on RunPod, fast pipe — never sync these from home):
   ```bash
   cd /workspace && mkdir -p models/checkpoints models/controlnet models/instantid \
     models/insightface/models/antelopev2
   pip install -q hf_transfer; export HF_HUB_ENABLE_HF_TRANSFER=1   # accelerator only — don't touch hub here

   huggingface-cli download SG161222/RealVisXL_V5.0 RealVisXL_V5.0_fp16.safetensors --local-dir models/checkpoints
   mv models/checkpoints/RealVisXL_V5.0_fp16.safetensors models/checkpoints/RealVisXL_V50_fp16.safetensors

   huggingface-cli download InstantX/InstantID ip-adapter.bin --local-dir models/instantid
   huggingface-cli download InstantX/InstantID ControlNetModel/diffusion_pytorch_model.safetensors --local-dir /tmp/iid
   mv /tmp/iid/ControlNetModel/diffusion_pytorch_model.safetensors models/controlnet/instantid-controlnet-sdxl.safetensors

   huggingface-cli download DIAMONIK7777/antelopev2 --local-dir models/insightface/models/antelopev2
   ls models/insightface/models/antelopev2/*.onnx   # must list 5 files, NOT nested
   ```
   > This pod just **downloads files** — the version pins (`hub<1.0`/`transformers<5`) are a
   > ComfyUI-*runtime* concern, handled in the serverless image (§3), not here. If `transformers`
   > is preinstalled (many templates ship 5.x, which wants `hub≥1.5`), do **not** downgrade hub on
   > this pod — just install `hf_transfer`. The fetched files are version-agnostic.
   > (CLI is `huggingface-cli download` on hub 0.x, `hf download` on hub 1.x — same args.)

   > **`/workspace/models` is the volume *root*, not a ComfyUI subdir.** This is intentional:
   > the store is shared by two ComfyUIs at *different* paths (dev pod
   > `…/runpod-slim/ComfyUI/models`, serverless `/comfyui/models`), so it can't live inside
   > either tree. Each ComfyUI points *into* the volume root — next step.

4. **Point each ComfyUI at the store** (don't move the models into a ComfyUI tree — redirect
   the tree to the models). Symlinking the whole `models/` dir covers the custom
   `instantid`/`insightface` folder types too, which `extra_model_paths.yaml` handles less cleanly.
   ComfyUI's path is **template-specific** (e.g. `/workspace/ComfyUI`,
   `/workspace/runpod-slim/ComfyUI`), so discover it rather than hardcoding:
   ```bash
   # DEV POD — find this template's ComfyUI, then wire its models dir to the volume root:
   COMFY=$(dirname "$(find / -maxdepth 6 -name main.py -path '*omfy*' 2>/dev/null | head -1)")
   echo "$COMFY"
   rm -rf "$COMFY/models" && ln -sfn /workspace/models "$COMFY/models"
   ```
   On **serverless** the worker's Dockerfile already does the equivalent
   (`ln -sfn /runpod-volume/models /comfyui/models`) — same volume, different mount point.

Decision: **models live on the volume root; ComfyUI + custom nodes live in the worker image** (§3),
each pointed at the store by a symlink. That hybrid gives you reproducible code/deps *and* one
shared model store the dev pod and serverless both read.

---

## 3. Build the pinned serverless worker image

This is the step that ends the dependency-hell. Everything is pinned at build time.

The files already live in the repo: **`comfy/serverless/Dockerfile`** and
**`comfy/serverless/send.py`**. The base tag is set to a current `-base` variant
(`runpod/worker-comfyui:5.8.5-base` — clean ComfyUI, no models; bump via the
[tags page](https://hub.docker.com/r/runpod/worker-comfyui/tags)). Set your Docker Hub user,
then build **from `comfy/`** so the context can see `custom_nodes/ComfyUI_FaceHash`:

```bash
cd "<repo>/comfy"
docker build -f serverless/Dockerfile -t <your-dockerhub-user>/facehash-worker:0.1 .
docker push <your-dockerhub-user>/facehash-worker:0.1
```

The Dockerfile is `FROM runpod/worker-comfyui:5.8.5-base` + `ComfyUI_InstantID` (cloned, its
`requirements.txt` installed → insightface/onnxruntime) + `ComfyUI_FaceHash` (copied) +
a `/comfyui/models` symlink to the volume. It deliberately **does not** re-pin
torch/transformers/huggingface_hub — the base image already ships a self-consistent set, and
overriding it is what causes drift. (FaceHash is numpy-only; nothing extra needed.)
A `comfy/.dockerignore` keeps the context lean.

> **Apple Silicon:** the base is **amd64-only** (RunPod is x86), so the Dockerfile pins
> `FROM --platform=linux/amd64`. On an M-series Mac it builds under emulation — slower, but the
> deps install from x86_64 wheels (no compile). The image must be amd64 to run on RunPod.
> If local emulation is too slow, let RunPod build from your GitHub repo instead, or build on an
> amd64 pod.
>
> No Docker locally? RunPod can build from a GitHub repo, or you can build on a throwaway pod.
> Verify the worker's ComfyUI path (`/comfyui` for these tags) and adjust the models symlink if
> a future base tag changes it.

---

## 4. Create the serverless endpoint

1. RunPod console → **Serverless → New Endpoint**.
2. **Container image:** `<your-dockerhub-user>/facehash-worker:0.1`.
3. **Network volume:** attach the one from §2 (it mounts at **`/runpod-volume`** here, *not*
   `/workspace`). Point ComfyUI at `/runpod-volume/models` — either via the worker's
   `extra_model_paths.yaml` or a symlink in your Dockerfile/handler:
   `ln -s /runpod-volume/models /comfyui/models`.
4. **GPU + scaling:** pick a GPU (24 GB is ample for SDXL); set **min workers 0** (scale to
   zero), max 1–2, an idle timeout (e.g. 5 s), and enable **FlashBoot** to cut cold starts.
5. Save. Note the **Endpoint ID** and your **API key** (Settings → API Keys).

---

## 5. Run a workflow from the Mac (inference)

1. **Generate the API payload** from canonical source:
   ```bash
   cd "<repo>/comfy" && python to_comfy_api.py     # -> comfy/api/*.json
   ```

2. **Send it** with `comfy/serverless/send.py` (real file in the repo — base64-encodes the
   photo, posts to `/runsync`, saves the returned image):
   ```bash
   export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
   python comfy/serverless/send.py comfy/api/M2_portrait_fp16.api.json \
       ~/photos/zack-normal-06.jpeg --key zacks-secret
   ```
   The photo's **basename must match** the `LoadImage` value baked in the workflow
   (`zack-normal-06.jpeg`) — the worker drops it into ComfyUI's input dir by that name.
   `--key` patches the FaceHash key for this run without rebuilding the workflow.

3. `/runsync` blocks and returns the result. For long jobs use `/run` (returns a job id) +
   poll `/status/<id>` or a webhook.

To change the **hash key** per request: pass `--key <newkey>` to `send.py` (it patches the
FaceHash node in place). To change the *default* baked into the workflow, edit `key` in
`build_m2.py` and re-run `to_comfy_api.py`.

---

## 6. The dev loop (occasional traditional pod)

When you need the **canvas** (trying nodes, debugging a graph), don't iterate on serverless:

1. Spin up a **GPU pod** (a ComfyUI template, or the `facehash-worker:0.1` image) with the
   **same network volume** attached (mounts `/workspace`).
2. **Wire nodes + models** — run **`comfy/pod_bootstrap.sh`** (keep a copy at
   `/workspace/pod_bootstrap.sh` so it's one command on every fresh pod). It (re)installs the
   InstantID deps, symlinks the FaceHash node, and points ComfyUI's `models/` at the volume
   store. Node *files* persist on the volume; the pip deps + symlinks it recreates do not.
   > **Mount-path gotcha:** the volume is **`/workspace` on a pod** but **`/runpod-volume` on
   > serverless** — the bootstrap symlinks the pod path; the serverless Dockerfile handles its own.
3. Restart ComfyUI, open `comfy/comfyui/*.json`, iterate.
4. When the graph is right: `to_comfy_api.py` → push → serverless.
5. **Stop the pod** — pods bill while running; the volume persists.

> **Fresh-pod CUDA snag:** if ComfyUI crashes with "NVIDIA driver too old," the host's driver is
> older than the template's torch (e.g. an L4 on a CUDA-12.8 host vs a cu130 template). Swap torch:
> `python3.12 -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

The pod is just the interactive face of the same volume-backed stack.

---

## 7. Syncing — the painless model

Split by data type; don't use one tool for everything.

### a) Workflows + node code → **git** (not Dropbox)

These are small text files that benefit from versioning. The repo's already in Dropbox at
home, but git is the clean transport *to RunPod*:

```bash
# Mac:  commit + push your generated workflows + node changes
git add comfy/ && git commit -m "update workflows" && git push

# Volume/pod:  pull
cd /workspace/face-hashing && git pull           # clone once: git clone <repo-url> first
```

Symlink the node from the cloned repo so edits flow with `git pull`:
`ln -s /workspace/face-hashing/comfy/custom_nodes/ComfyUI_FaceHash /comfyui/custom_nodes/`.

### b) Models → **pull from HuggingFace on the volume** (don't sync from home)

Covered in §2. Zero egress, fast. Re-running the `huggingface-cli download` lines is the
"sync" — idempotent.

### c) Private binaries (input photos, outputs) → **rclone**

This is where your Dropbox intuition is right — but for the *binaries*, not the workflows.
Since the repo already lives in Dropbox, an **rclone Dropbox remote** is the natural pipe:

```bash
pip install rclone || curl https://rclone.org/install.sh | sudo bash
rclone config            # new remote "dropbox", type=dropbox, OAuth in browser once
# pull personal inputs onto the pod/volume:
rclone copy dropbox:"Team Folder/.../face-hashing/local/in" /workspace/in -P
# push results back home:
rclone copy /workspace/out dropbox:"Team Folder/.../face-hashing/runpod-out" -P
```

Caveats: the OAuth step is once-per-machine; transfer speed is your home/Dropbox link, which
is fine for a handful of photos but don't route *models* through it. Alternative (no Dropbox
OAuth): the Tailscale + `rclone copy` over SFTP path from earlier — pick whichever you find
less annoying. **Use `rclone copy` (not `mount`)** on serverless/pods — RunPod restricts FUSE.

> Rule of thumb: **git for code, HF for models, rclone/Dropbox for private binaries.** Keep
> personal photos out of git (they already are, via `.gitignore`).

---

## 8. Cost hygiene / teardown

- **Serverless** scales to zero — you pay per request-second only. FlashBoot trims cold starts.
- **Dev pods** bill while running — stop them when you stop iterating.
- **Network volume** bills ~$0.07/GB/mo while it exists, even idle. Keep it for your model set;
  delete when a project's done.

---

## Appendix: pinned versions & the traps we already hit

- **The principle: `huggingface_hub` must match the `transformers` of the ComfyUI you *run*.**
  hub 1.0 was a breaking release — `transformers 4.x` needs `hub<1.0`, `transformers 5.x` needs
  `hub>=1.5`. Mismatches crash ComfyUI at `from transformers import CLIPTokenizer`. This cuts
  *both* ways: a 4.x stack breaks if hub jumps to 1.x, and a 5.x stack (e.g. the `comfyui-cuda-13`
  template, transformers 5.7.0) breaks if you downgrade hub below 1.0. Only downgrade hub on a
  pod you actually run ComfyUI on, to match *its* transformers.
- **The serverless worker** (`runpod/worker-comfyui:*-base`) already ships a self-consistent
  torch/transformers/huggingface_hub set. The Dockerfile **adds only InstantID's deps** and
  leaves that stack alone — overriding a clean pinned base is what causes drift, not what
  prevents it. (Earlier drafts pinned `transformers<5`/`hub<1.0` here; that was wrong for this
  base and has been removed.)
- **Download-only pods** don't care about any of this — install `hf_transfer`, leave hub alone;
  the fetched files are version-agnostic.
- **mmap loader** — if you ever see `'ModelMMAP' object has no attribute 'get_file_handle'`,
  launch ComfyUI with **`--disable-mmap`** (stale `comfy_aimdo` vs core). A pinned image avoids it.
- **antelopev2** — the 5 `.onnx` files must sit **directly** in
  `models/insightface/models/antelopev2/`, not a nested `antelopev2/antelopev2/`.
- **InstantIDFaceAnalysis = CPU** in our workflow → CPU `onnxruntime` is fine; no GPU-onnx needed.

## Quick reference — the whole loop once it's set up

```bash
# design (Mac or dev pod) ▸ regenerate ▸ push ▸ infer
python comfy/to_comfy_api.py
git add comfy && git commit -m wf && git push
python comfy/serverless/send.py comfy/api/M2_portrait_fp16.api.json ~/photos/zack-normal-06.jpeg
```
