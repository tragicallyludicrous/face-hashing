"""build_gallery.py — build the real-identity gallery for `arcface_blend_v2`.

The Stage-2 hash `arcface_blend_v2` blends a face toward a key-selected REAL identity so the
result stays on the real-face manifold (a generator like InstantID then renders a plausible
human for any input — unlike the off-manifold signed permutation). This script assembles that
gallery from a folder of real face photos.

It writes a **paired** gallery (one row per face, row-aligned across two backbones):

    antelope : (G, 512)  insightface antelopev2 recognition embeddings  -> the InstantID texture path
    mica     : (G, 512)  MICA's frozen-ArcFace embeddings               -> the MICA depth/geometry path
    names    : (G,)      source filenames

Because the rows are aligned, the SAME key selects the SAME real person in BOTH spaces, so the
hashed geometry (mica column) and hashed texture (antelope column) depict ONE synthetic identity
— the shared-identity refactor — even though MICA and antelopev2 are different ArcFace models.

    python build_gallery.py -i faces/ -o gallery.npz                 # paired (needs MICA)
    python build_gallery.py -i faces/ -o gallery.npz --no-mica       # antelope only (texture)

Use real, rights-cleared faces (a public face dataset / stock). Keep galleries out of git if the
faces are sensitive. A few hundred varied identities is plenty.
"""
import argparse
import os

import numpy as np

import mica_local as mica
import hash_shape                       # reuse its detect+align+MICA-encode (_codedict)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def main():
    ap = argparse.ArgumentParser(description="Build the real-identity gallery for arcface_blend_v2.")
    ap.add_argument("-i", "--in-dir", required=True, help="folder of real face photos")
    ap.add_argument("-o", "--out", default="gallery.npz", help="output .npz (paired) / .npy (antelope only)")
    ap.add_argument("--no-mica", action="store_true", help="antelope column only (skip MICA; texture-only)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import cv2
    h = mica.load(device=a.device)       # h.app == insightface antelopev2 (detect + recognition)

    paths = []
    for root, _, files in os.walk(a.in_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in EXTS:
                paths.append(os.path.join(root, fn))
    if not paths:
        raise SystemExit(f"no images under {a.in_dir}")

    antelope, micacol, names = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  skip (unreadable): {p}"); continue
        face = _largest_face(h.app.get(img))            # antelopev2 full pipeline
        if face is None:
            print(f"  skip (no face):    {os.path.basename(p)}"); continue
        a_emb = np.asarray(face.normed_embedding, np.float32)   # (512,) antelopev2 recognition

        m_emb = None
        if not a.no_mica:
            got = hash_shape._codedict(h, p)            # (codedict, raw 512 MICA arcface)
            if got is None:
                print(f"  skip (mica no face): {os.path.basename(p)}"); continue
            m_emb = got[1].astype(np.float32)

        antelope.append(a_emb)
        if m_emb is not None:
            micacol.append(m_emb)
        names.append(os.path.basename(p))
        print(f"  + {os.path.basename(p)}")

    if not antelope:
        raise SystemExit("no usable faces — gallery empty")
    A = np.stack(antelope)
    names = np.array(names)
    if a.no_mica or not micacol:
        if a.out.endswith(".npz"):
            np.savez(a.out, antelope=A, names=names)
        else:
            np.save(a.out, A)
        print(f"\n{len(A)} faces -> {a.out}  (antelope only)")
    else:
        M = np.stack(micacol)
        assert len(A) == len(M) == len(names), "paired columns must be row-aligned"
        np.savez(a.out, antelope=A, mica=M, names=names)
        print(f"\n{len(A)} faces -> {a.out}  (paired: antelope {A.shape}, mica {M.shape})")


if __name__ == "__main__":
    main()
