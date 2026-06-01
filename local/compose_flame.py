"""
compose_flame.py — MICA identity + DECA expression/pose -> a FLAME .glb (Stage 3, "reconstruct").

Decodes flame(shape = MICA identity.npy, exp + pose = DECA <stem>_params.npz) so you can SEE a
(possibly hashed) identity wearing the ORIGINAL photo's expression and head pose. MICA supplies the
consistent identity; DECA supplies expression/pose; they share the FLAME 2020 basis.

    python compose_flame.py <identity.npy> <deca_params.npz> -o out.glb

e.g.
    python compose_flame.py out/emma-restaurant/identity.npy \
        "$HOME/Library/CloudStorage/.../My Drive/Face-Hashing/Output/emma-restaurant/emma-restaurant_params.npz" \
        -o out/emma-restaurant/composed.glb

Then drop the .glb into viewer/models/ and serve the viewer to drag-rotate it. Use the SAME stem in
both paths (same person+photo) for a sensible result. The identity.npy can be a raw MICA one (sanity
check: should roughly reproduce the original posed face) or a hashed one (the swapped identity).
"""
import argparse

import mica_local

ap = argparse.ArgumentParser(description="Compose MICA identity + DECA expression/pose into a FLAME .glb.")
ap.add_argument("identity", help="MICA identity.npy (300-d FLAME shape; raw or hashed)")
ap.add_argument("deca_npz", help="DECA <stem>_params.npz (uses its 'exp' and 'pose')")
ap.add_argument("-o", "--out", required=True, help="output .glb path")
ap.add_argument("--device", default="cpu", help="cpu (default) | mps | auto")
a = ap.parse_args()

h = mica_local.load(device=a.device, detector="none")     # FLAME decoder only — no face detection
mica_local.compose_mesh(h, a.identity, a.deca_npz, a.out)
