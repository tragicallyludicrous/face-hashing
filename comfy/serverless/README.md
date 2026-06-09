# comfy/serverless

Pinned RunPod serverless worker for Stage-4 inference. Full walkthrough:
[`../../runpod-serverless-migration.md`](../../runpod-serverless-migration.md).

- **`Dockerfile`** — `FROM runpod/worker-comfyui` + pinned deps + `ComfyUI_InstantID` +
  `ComfyUI_FaceHash`. Fill in the `<tag>` and your Docker Hub user.
- **`send.py`** — POST a `comfy/api/*.api.json` workflow (+ a base64 photo) to your endpoint.

## Build & push

```bash
cd ..                                              # the comfy/ dir (build context)
docker build -f serverless/Dockerfile -t <user>/facehash-worker:0.1 .
docker push <user>/facehash-worker:0.1
```

## Run

```bash
python ../to_comfy_api.py                          # regenerate comfy/api/*.json from build_*.py
export RUNPOD_ENDPOINT_ID=...  RUNPOD_API_KEY=...
python send.py ../api/M2_portrait_fp16.api.json ~/photos/zack-normal-06.jpeg --key zacks-secret
```

Models live on the attached **network volume** (`/runpod-volume/models`), not in the image —
see §2 of the manual for populating it.
