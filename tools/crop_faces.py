"""
crop_faces.py — detect each local/in photo's face box and write local/out/crops.json, so
viewer/studio.html can crop the input-photo thumbnail to the subject's head.

Reuses the antelopev2/RetinaFace detector already on disk (~/.insightface/models, the one the
pipeline uses). Detection only — no MICA/SMIRK, so it's fast. Run once after adding/renaming photos:

    local/.venv/bin/python tools/crop_faces.py

Output (git-ignored, under local/out): {"<stem>": {"x":, "y":, "w":, "h":}} — the face box in
NORMALIZED [0,1] image coords. The viewer pads/centers it into a head crop, so this stays the raw
detection (re-tune the framing in the viewer without re-detecting).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "local", "in")
OUT = os.path.join(ROOT, "local", "out", "crops.json")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="antelopev2", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))           # larger det_size -> catches small-in-frame faces

    crops, missed = {}, []
    for f in sorted(os.listdir(IN)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in EXTS:
            continue
        img = cv2.imread(os.path.join(IN, f))
        if img is None:
            missed.append(stem); continue
        H, W = img.shape[:2]
        faces = app.get(img)
        if not faces:
            missed.append(stem); continue
        x1, y1, x2, y2 = max(faces, key=lambda x: x.det_score).bbox   # most-confident face
        crops[stem] = {"x": float(x1 / W), "y": float(y1 / H),
                       "w": float((x2 - x1) / W), "h": float((y2 - y1) / H)}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(crops, open(OUT, "w"), indent=0)
    print(f"wrote {OUT}: {len(crops)} face box(es)" + (f"; no face: {', '.join(missed)}" if missed else ""))


if __name__ == "__main__":
    main()
