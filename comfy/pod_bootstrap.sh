#!/usr/bin/env bash
# Re-wire Face-Hashing custom nodes + models on a fresh / migrated RunPod. Idempotent — run it
# after every pod bring-up, then restart ComfyUI. Covers both texture paths: InstantID (SDXL) and
# PuLID + Flux.2 Klein (the Flux MVP).
#
# Node *files* persist on the network volume; pip *deps* and symlinks live in the ephemeral
# container, so they need recreating each fresh pod.
#
# One-time on a brand-new volume:
#   - InstantID / PuLID-Flux2 nodes: cloned automatically (public).
#   - FaceHash node: clone your repo to /workspace/face-hashing (PAT), OR unzip ComfyUI_FaceHash
#                    into $COMFY/custom_nodes/. It then persists on the volume.
#   - Weights: export HF_TOKEN (a read token whose account has ACCEPTED the gated licenses — see
#              FLUX2_PULID_GETTING_STARTED.md §2). Gated Klein weights then download to the volume
#              and persist; without the token, present files are kept and missing ones are reported.
#
# Not handled here (host-specific): a torch/driver mismatch. On Blackwell (RTX PRO 4500, sm_120)
# you need a cu128 torch — if renders come out pure noise, swap torch for a cu128 build first.
#
# Usage:  [HF_TOKEN=hf_xxx] bash /workspace/face-hashing/comfy/pod_bootstrap.sh
set -e

# --- Locate ComfyUI + the EXACT python it runs with (venv-aware). Do NOT use
#     `readlink -f /proc/$PID/exe`: it resolves a venv's symlinked python to the BASE interpreter,
#     so pip would install into the wrong env and the node fails to import / errors at runtime. ---
PID=$(pgrep -f '[m]ain.py' | head -1 || true)
COMFY="${COMFY:-$( { [ -n "$PID" ] && readlink "/proc/$PID/cwd"; } || echo /workspace/runpod-slim/ComfyUI )}"

find_py() {
  if [ -n "$PID" ]; then
    local v a
    v=$(tr '\0' '\n' < "/proc/$PID/environ" 2>/dev/null | sed -n 's/^VIRTUAL_ENV=//p')
    [ -n "$v" ] && [ -x "$v/bin/python" ] && { echo "$v/bin/python"; return 0; }
    a=$(tr '\0' '\n' < "/proc/$PID/cmdline" 2>/dev/null | head -1)
    case "$a" in */*) [ -x "$a" ] && { echo "$a"; return 0; };; esac
  fi
  local c
  for c in "${VIRTUAL_ENV:-}/bin/python" "$COMFY/venv/bin/python" /workspace/venv/bin/python; do
    [ -x "$c" ] && "$c" -c 'import torch' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
PY=${PY:-$(find_py)} || true
[ -n "${PY:-}" ] || { echo "!! Can't find ComfyUI's venv python. Start ComfyUI once, or re-run with PY=/path/to/venv/bin/python."; exit 1; }
echo "ComfyUI: $COMFY"
"$PY" -c 'import sys,torch; print("python:",sys.executable,"| torch",torch.__version__)' \
  || { echo "!! $PY can't import torch — wrong interpreter; re-run with PY= set explicitly."; exit 1; }

# --- Shared face deps (InstantID + PuLID both need these); onnxruntime-GPU for the pod. ---
"$PY" -m pip install -q insightface onnxruntime-gpu open-clip-torch safetensors
# ml_dtypes >= 0.5.1 — NOT the PuLID README's 0.3.2. The old pin lacks `float4_e2m1fn` (the FP4
# dtype a Flux.2 / Blackwell torch needs), which makes InsightFace fail to load at RUN time. Verify.
"$PY" -m pip install -q -U "ml_dtypes>=0.5.1"
"$PY" -c 'import ml_dtypes,sys; print("ml_dtypes",ml_dtypes.__version__,"float4",hasattr(ml_dtypes,"float4_e2m1fn")); sys.exit(0 if hasattr(ml_dtypes,"float4_e2m1fn") else 1)' \
  || { echo "!! ml_dtypes still lacks float4_e2m1fn — something re-pinned it. Inspect: $PY -m pip show ml_dtypes onnxruntime-gpu"; exit 1; }

# --- Custom nodes (clone only if absent; they persist on the volume) ---
# InstantID: manager install names it `comfyui_instantid`, a git clone `ComfyUI_InstantID` —
# don't clone a second copy. (FaceHash imports whichever exists.)
if [ ! -d "$COMFY/custom_nodes/ComfyUI_InstantID" ] && [ ! -d "$COMFY/custom_nodes/comfyui_instantid" ]; then
  git clone https://github.com/cubiq/ComfyUI_InstantID.git "$COMFY/custom_nodes/ComfyUI_InstantID"
fi
# PuLID + Flux.2 (the Flux MVP)
[ -d "$COMFY/custom_nodes/ComfyUI-PuLID-Flux2" ] \
  || git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git "$COMFY/custom_nodes/ComfyUI-PuLID-Flux2"
# FaceHash (Stage-2 hash; persists on the volume once present)
if [ -e "$COMFY/custom_nodes/ComfyUI_FaceHash/nodes.py" ]; then
  echo "FaceHash present."
elif [ -d /workspace/face-hashing ]; then
  ln -sfn /workspace/face-hashing/comfy/custom_nodes/ComfyUI_FaceHash "$COMFY/custom_nodes/ComfyUI_FaceHash"
else
  echo "!! FaceHash missing — unzip ComfyUI_FaceHash.zip into $COMFY/custom_nodes/ (or clone your repo to /workspace/face-hashing)"
fi

# --- model store: ComfyUI's own models/ (it's on the persistent volume, so it already survives a
#     migrate — no symlink needed). Set MODELS= only if you keep a separate/shared store. ---
MODELS="${MODELS:-$COMFY/models}"
mkdir -p "$MODELS"

# --- Flux.2 Klein weights (gated; fetch only if missing; needs HF_TOKEN + accepted license) ---
get() { # get <subdir> <file> <url>
  local d="$MODELS/$1"; mkdir -p "$d"
  if [ -s "$d/$2" ]; then echo "  have $1/$2"; return 0; fi
  if [ -z "${HF_TOKEN:-}" ]; then echo "  !! missing $1/$2 — export HF_TOKEN (license-accepted read token) and re-run, or copy it onto the volume"; return 0; fi
  echo "  fetching $1/$2 ..."
  curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$d/$2" "$3" \
    || echo "  !! download failed for $1/$2 — license accepted on that repo? token valid?"
}
HF=https://huggingface.co
get diffusion_models flux-2-klein-9b-fp8.safetensors  $HF/Comfy-Org/flux2-klein-9B/resolve/main/split_files/diffusion_models/flux-2-klein-9b-fp8.safetensors
get text_encoders    qwen_3_8b_fp8mixed.safetensors   $HF/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
get vae              flux2-vae.safetensors            $HF/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors
get pulid            pulid_flux2_klein_v2.safetensors $HF/Fayens/Pulid-Flux2/resolve/main/pulid_flux2_klein_v2.safetensors
[ -e "$MODELS/insightface/models/antelopev2/glintr100.onnx" ] \
  || echo "  !! antelopev2 not in $MODELS/insightface/models/antelopev2/ — PuLID & InstantID need it; see FLUX2_PULID_GETTING_STARTED.md §2"

# --- canvas workflows <-> git: Save in the canvas writes into the repo, and a fresh pod's
#     `git pull` repopulates the sidebar (FLUX2_pulid_roundtrip.json lives in comfy/workflows/). ---
if [ -d /workspace/face-hashing ]; then
  mkdir -p /workspace/face-hashing/comfy/workflows "$COMFY/user/default"
  ln -sfn /workspace/face-hashing/comfy/workflows "$COMFY/user/default/workflows"
fi

echo "done — restart ComfyUI (Manager > Restart, or relaunch main.py) to load the nodes."
echo "  Flux.2 MVP workflow appears in the canvas sidebar: FLUX2_pulid_roundtrip.json"
# Push/pull the repo from the pod: set a credential ONCE (persists on the volume):
#   git config --global credential.helper store   # then `git pull` caches your GitHub user + PAT
