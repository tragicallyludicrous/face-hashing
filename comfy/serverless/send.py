"""POST a ComfyUI API-format workflow to a RunPod serverless endpoint and save the
returned image(s). Pairs with comfy/to_comfy_api.py (which generates the workflow).

  export RUNPOD_ENDPOINT_ID=...  RUNPOD_API_KEY=...
  python comfy/serverless/send.py comfy/api/M2_portrait_fp16.api.json ~/photos/zack-normal-06.jpeg [--key zacks-secret]

The image filename MUST match the LoadImage node's value in the workflow — the worker
drops it into ComfyUI's input dir by that name. Use --key to override the FaceHash key
without rebuilding the workflow (patches the FaceHashApplyInstantID node in place).
"""

import argparse
import base64
import json
import os

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")                 # comfy/api/*.api.json
    ap.add_argument("image")                    # local photo; basename must match LoadImage
    ap.add_argument("--key", help="override the FaceHash key for this run")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    ep = os.environ["RUNPOD_ENDPOINT_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]

    wf = json.load(open(args.workflow))
    if args.key is not None:                    # patch FaceHashApplyInstantID.key in place
        for node in wf.values():
            if node.get("class_type") == "FaceHashApplyInstantID":
                node["inputs"]["key"] = args.key

    name = os.path.basename(args.image)
    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    payload = {"input": {"workflow": wf, "images": [{"name": name, "image": b64}]}}

    r = requests.post(
        f"https://api.runpod.ai/v2/{ep}/runsync",
        headers={"Authorization": api_key},
        json=payload,
        timeout=args.timeout,
    )
    r.raise_for_status()
    data = r.json()

    out = (data.get("output") or {}).get("images", [])
    if not out:
        print("no images returned; full response:")
        print(json.dumps(data, indent=2)[:2000])
        return
    for i, img in enumerate(out):
        if img.get("type") == "base64":
            path = f"out_{i}.png"
            open(path, "wb").write(base64.b64decode(img["data"]))
            print("wrote", path)
        else:
            print("s3:", img.get("data"))


if __name__ == "__main__":
    main()
