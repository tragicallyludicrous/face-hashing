"""
export_flame_basis.py — dump FLAME's SHAPE basis to a browser-loadable binary for viewer/sliders.html.

FLAME identity is a LINEAR blendshape model:  vertices = v_template + shapedirs[:, :, :K] @ beta.
This exports v_template, the first K shape blendshape directions, and the faces, so the Three.js
viewer can evaluate that sum live as you drag sliders — and seed `beta` directly from a MICA
identity.npy (MICA's 300-d output IS FLAME shape beta), reproducing that person's neutral face.

LICENSE: FLAME is registration-gated / non-commercial. Output goes to viewer/flame/ (git-ignored).
Do NOT commit it and do NOT serve the viewer publicly — that would redistribute FLAME weights.

    # from the repo root, with the local venv (it has chumpy, which the pkl needs):
    local/.venv/bin/python tools/export_flame_basis.py
    local/.venv/bin/python tools/export_flame_basis.py --pkl /path/to/generic_model.pkl --n-shape 300
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PKL = os.path.join(ROOT, "local", "MICA", "data", "FLAME2020", "generic_model.pkl")
OUTDIR = os.path.join(ROOT, "viewer", "flame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL, help="FLAME generic_model.pkl")
    ap.add_argument("--n-shape", type=int, default=300, help="shape PCs to export (MICA uses 300)")
    a = ap.parse_args()
    if not os.path.exists(a.pkl):
        sys.exit(f"FLAME model not found: {a.pkl}\n(point --pkl at your generic_model.pkl)")

    with open(a.pkl, "rb") as f:                 # pkl holds chumpy arrays -> needs chumpy importable
        m = pickle.load(f, encoding="latin1")

    v_template = np.asarray(m["v_template"], dtype=np.float32)        # (nv, 3)
    shapedirs = np.asarray(m["shapedirs"], dtype=np.float32)         # (nv, 3, 400): [:, :, :300]=shape
    faces = np.asarray(m["f"], dtype=np.uint32)                     # (nf, 3)

    nv = v_template.shape[0]
    K = min(a.n_shape, shapedirs.shape[2])
    # row i = shape blendshape i, flattened vertex-major to match Three.js positions [x0,y0,z0,...]
    S = np.transpose(shapedirs[:, :, :K], (2, 0, 1)).reshape(K, nv * 3).astype(np.float32)

    os.makedirs(OUTDIR, exist_ok=True)
    bin_path = os.path.join(OUTDIR, "flame_basis.bin")
    with open(bin_path, "wb") as f:
        v_template.reshape(-1).astype("<f4").tofile(f)     # [0]              3*nv float32
        S.reshape(-1).astype("<f4").tofile(f)              # [after vtempl]   K*3*nv float32
        faces.reshape(-1).astype("<u4").tofile(f)          # [after dirs]     3*nf uint32
    json.dump({"n_verts": int(nv), "n_faces": int(faces.shape[0]), "n_shape": int(K),
               "order": ["v_template f4[3*n_verts]", "shapedirs f4[n_shape*3*n_verts]",
                         "faces u4[3*n_faces]"]},
              open(os.path.join(OUTDIR, "flame_basis.json"), "w"), indent=2)

    mb = os.path.getsize(bin_path) / 1e6
    print(f"wrote {bin_path} ({mb:.1f} MB): {nv} verts, {faces.shape[0]} faces, {K} shape PCs")
    print("git-ignored (FLAME license). Serve locally:  cd viewer && python3 -m http.server 8080  -> /sliders.html")


if __name__ == "__main__":
    main()
