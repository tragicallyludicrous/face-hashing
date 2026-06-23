"""
hash_shape.py — Phase B core: photo + key -> HASHED 300-d MICA shape (.npy).

MICA maps an ArcFace embedding -> 300-d FLAME shape. We hash the embedding with the SAME
`arcface_keymix_v1` the FaceHash ComfyUI node uses, BETWEEN MICA's encode and decode, so the
resulting shape is a deterministic-but-different head per key — the geometry analog of the
InstantID texture hash. Feed the .npy to `render_cond.py --shape` to get the posed depth.

    python hash_shape.py in/zack-normal-04.jpeg --keys raw zack-secret another-key -o bridge_test/hashed

Runs in the MICA interpreter (separate from SMIRK/render_cond), like run.py's subprocess split.
"""
import argparse
import os

import numpy as np
import torch

import mica_local as mica
from arcface_hash import arcface_keymix_v1, arcface_blend_v2, load_gallery, cosine


def _codedict(h, image_path):
    """Replicate mica._run up to encode; return (codedict, raw 512 embedding) or None."""
    import cv2
    from insightface.app.common import Face
    from datasets.creation.util import get_arcface_input, get_center

    img = cv2.imread(image_path)
    if img is None:
        return None
    bboxes, kpss = h.app.detect(img)
    if bboxes is None or len(bboxes) == 0:
        return None
    i = get_center(bboxes, img)
    face = Face(bbox=bboxes[i, 0:4], kps=(kpss[i] if kpss is not None else None), det_score=bboxes[i, 4])
    blob, aimg = get_arcface_input(face, img)
    arcface_img = torch.tensor(blob).float().to(h.device)[None]            # (1,3,112,112)
    images = cv2.resize(aimg, (224, 224)).transpose(2, 0, 1) / 255.0
    images = torch.tensor(images).float().to(h.device)[None]              # (1,3,224,224) unused for ID
    with torch.no_grad():
        codedict = h.mica.encode(images, arcface_img)
    raw = codedict["arcface"][0].detach().cpu().numpy().astype(np.float32)  # (512,) L2-norm
    return codedict, raw


def shape_for(h, codedict, raw_emb, transform_fn):
    """Decode a (possibly hashed) embedding -> 300-d shape. transform_fn: raw 512 -> emb 512."""
    emb = np.asarray(transform_fn(raw_emb), dtype=np.float32)
    cd = dict(codedict)
    cd["arcface"] = torch.tensor(emb)[None].to(h.device)
    with torch.no_grad():
        opdict = h.mica.decode(cd)
    return opdict["pred_shape_code"][0].detach().cpu().numpy().astype(np.float32)


def _transform_for(key, args, gallery):
    """Build raw512 -> emb512 for one key. key in ('','raw') -> identity (un-hashed)."""
    if key in ("", "raw", None):
        return lambda e: np.asarray(e, np.float32)
    if args.transform == "arcface_blend_v2":
        return lambda e: arcface_blend_v2(e, key, gallery, args.strength, args.n_mix)
    return lambda e: arcface_keymix_v1(e, key, args.offset)


def main():
    ap = argparse.ArgumentParser(description="Photo + key -> hashed 300-d MICA shape .npy (Phase B).")
    ap.add_argument("image", help="input photo")
    ap.add_argument("--keys", nargs="+", default=["raw", "zack-secret"], help="keys ('raw' = un-hashed)")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--transform", choices=["arcface_keymix_v1", "arcface_blend_v2"],
                    default="arcface_keymix_v1")
    ap.add_argument("--gallery", help="paired gallery .npz (required for arcface_blend_v2)")
    ap.add_argument("--column", default="mica", help="gallery column for the MICA geometry path")
    ap.add_argument("--strength", type=float, default=0.6, help="blend_v2: 0=self .. 1=target id")
    ap.add_argument("--n-mix", dest="n_mix", type=int, default=2, help="blend_v2: gallery rows/key")
    ap.add_argument("-o", "--out-dir", default="bridge_test/hashed")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    gallery = None
    if a.transform == "arcface_blend_v2":
        if not a.gallery:
            raise SystemExit("arcface_blend_v2 needs --gallery (build one with build_gallery.py)")
        gallery = load_gallery(a.gallery, a.column)

    h = mica.load(device=a.device)
    got = _codedict(h, a.image)
    if got is None:
        raise SystemExit(f"no face in {a.image}")
    codedict, raw = got

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    shapes = {}
    for k in a.keys:
        s = shape_for(h, codedict, raw, _transform_for(k, a, gallery))
        tag = "raw" if k in ("", "raw") else k
        path = os.path.join(a.out_dir, f"{stem}__{tag}.npy")
        np.save(path, s)
        shapes[tag] = s
        print(f"{tag:>14}: shape300 |mean|={np.abs(s).mean():.3f} max|{np.abs(s).max():.2f}|  -> {path}")

    # how different are the hashed identities from each other / from raw? (cosine in shape space)
    tags = list(shapes)
    print("\nshape-space cosine (lower = more different identity):")
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            print(f"  {tags[i]:>14} vs {tags[j]:<14} {cosine(shapes[tags[i]], shapes[tags[j]]):+.3f}")


if __name__ == "__main__":
    main()
