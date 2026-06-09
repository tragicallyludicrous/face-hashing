# Pinned ComfyUI serverless worker for Face-Hashing (Stage 4).
#
# Build from the comfy/ dir so the context can see custom_nodes/:
#   cd <repo>/comfy
#   docker build -f serverless/Dockerfile -t <dockerhub-user>/facehash-worker:0.1 .
#   docker push <dockerhub-user>/facehash-worker:0.1
#
# Models are NOT baked in — they come from the attached network volume at
# /runpod-volume/models (symlinked below). Base tag is the "-base" variant
# (clean ComfyUI, no models). Newer tags: https://hub.docker.com/r/runpod/worker-comfyui/tags
# If you hit a CUDA/GPU mismatch on a very new card, try the *-base-cudaXX.Y.Z tag.
#
# --platform=linux/amd64 is REQUIRED: RunPod GPUs are x86 and this base is amd64-only,
# so on an Apple-Silicon (arm64) Mac it builds under emulation. That's expected — the
# resulting image must be amd64 to run on RunPod.
FROM --platform=linux/amd64 runpod/worker-comfyui:5.8.5-base

# Add ONLY InstantID's extra deps. The base image already ships a self-consistent
# torch/transformers/huggingface_hub set — do NOT override it. (The version fights
# that broke runpod-slim came from reconciling a fragile shared env; this image is
# pinned and reproducible, so leaving its core stack untouched IS the fix.)
# FaceHash itself is numpy-only and needs nothing extra.
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git && \
    pip install --no-cache-dir -r ComfyUI_InstantID/requirements.txt
# (reproducibility: pin a commit ->  && cd ComfyUI_InstantID && git checkout <sha>)
COPY custom_nodes/ComfyUI_FaceHash ./ComfyUI_FaceHash

# Point ComfyUI at the network-volume model store (mounted /runpod-volume on serverless).
RUN ln -sfn /runpod-volume/models /comfyui/models
