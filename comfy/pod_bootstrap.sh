#!/usr/bin/env bash
# Re-wire Face-Hashing custom nodes + models on a fresh RunPod dev pod. Idempotent.
#
# Node *files* persist on the network volume (custom_nodes/ under the volume-backed
# ComfyUI); the pip *deps* and symlinks live in the ephemeral container, so they need
# recreating each fresh pod. Run this after every deploy, then restart ComfyUI.
#
# One-time on a brand-new volume:
#   - InstantID: cloned automatically (public).
#   - FaceHash : clone your repo to /workspace/face-hashing (PAT), OR unzip
#                ComfyUI_FaceHash into $COMFY/custom_nodes/. It then persists.
#
# Not handled here (host-specific): a torch/driver mismatch (e.g. L4 with an older
# CUDA-12.8 driver vs a cu130 template) — fix that separately with a cu128 torch swap.
#
# Usage:  bash /workspace/pod_bootstrap.sh
set -e

# Derive ComfyUI dir + its python from the running process; fall back to known defaults.
PID=$(pgrep -f 'main.py' | head -1)
COMFY=$(readlink "/proc/$PID/cwd" 2>/dev/null || echo /workspace/runpod-slim/ComfyUI)
PY=$(readlink "/proc/$PID/exe" 2>/dev/null || echo /usr/bin/python3.12)
echo "ComfyUI: $COMFY   python: $PY"

# 1. InstantID node + deps. Clone only if NEITHER name is present — a ComfyUI-Manager
#    install names the folder `comfyui_instantid`, a git clone names it `ComfyUI_InstantID`;
#    cloning a second copy would duplicate the nodes. (FaceHash imports whichever exists.)
if [ ! -d "$COMFY/custom_nodes/ComfyUI_InstantID" ] && [ ! -d "$COMFY/custom_nodes/comfyui_instantid" ]; then
  git clone https://github.com/cubiq/ComfyUI_InstantID.git "$COMFY/custom_nodes/ComfyUI_InstantID"
fi
"$PY" -m pip install -q insightface onnxruntime

# 2. FaceHash node (persists on the volume once present)
if [ -e "$COMFY/custom_nodes/ComfyUI_FaceHash/nodes.py" ]; then
  echo "FaceHash present."
elif [ -d /workspace/face-hashing ]; then
  ln -sfn /workspace/face-hashing/comfy/custom_nodes/ComfyUI_FaceHash "$COMFY/custom_nodes/ComfyUI_FaceHash"
else
  echo "!! FaceHash missing — unzip ComfyUI_FaceHash.zip into $COMFY/custom_nodes/ (or clone your repo to /workspace/face-hashing)"
fi

# 3. models -> volume store
mkdir -p /workspace/models
ln -sfn /workspace/models "$COMFY/models"

# 4. canvas workflows <-> git: make ComfyUI's workflows dir a repo folder, so Save in the
#    canvas writes into the repo and a fresh pod's `git pull` repopulates the sidebar.
if [ -d /workspace/face-hashing ]; then
  mkdir -p /workspace/face-hashing/comfy/workflows "$COMFY/user/default"
  ln -sfn /workspace/face-hashing/comfy/workflows "$COMFY/user/default/workflows"
fi

echo "done — restart ComfyUI (Manager > Restart, or relaunch main.py) to load the nodes."
# To push/pull the repo from the pod, set a credential ONCE (persists on the volume):
#   git config --global credential.helper store
#   then a `git pull` will prompt for your GitHub username + a PAT, and cache it.
