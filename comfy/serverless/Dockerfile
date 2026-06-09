# Pinned ComfyUI serverless worker for Face-Hashing (Stage 4).
#
# Build from the comfy/ dir so the context can see custom_nodes/:
#   cd <repo>/comfy
#   docker build -f serverless/Dockerfile -t <dockerhub-user>/facehash-worker:0.1 .
#   docker push <dockerhub-user>/facehash-worker:0.1
#
# Models are NOT baked in — they come from the attached network volume at
# /runpod-volume/models (symlinked below). Pick a current base tag from
#   https://hub.docker.com/r/runpod/worker-comfyui/tags
FROM runpod/worker-comfyui:<tag>

# Pinned deps — these exact bounds are what stop ComfyUI from breaking
# (huggingface_hub 1.0 / transformers 5.x crash). Do NOT loosen them.
RUN pip install --no-cache-dir \
      "huggingface_hub<1.0" "transformers<5" "tokenizers<0.22" \
      insightface onnxruntime

# Custom nodes — InstantID FIRST (FaceHash subclasses it and imports it by name).
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git && \
    pip install --no-cache-dir -r ComfyUI_InstantID/requirements.txt
# (reproducibility: pin a commit ->  && cd ComfyUI_InstantID && git checkout <sha>)
COPY custom_nodes/ComfyUI_FaceHash ./ComfyUI_FaceHash

# Use the network-volume model store (mounted at /runpod-volume on serverless).
# Verify the ComfyUI path for your base tag (usually /comfyui); adjust if needed.
RUN ln -sfn /runpod-volume/models /comfyui/models
