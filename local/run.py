"""
run.py — one command, end-to-end local pipeline: photos -> per-person {arcface.npy, composed.glb}.

    python run.py -i in -o out

For every image in <in>, it writes out/<stem>/ containing exactly two files:
  1. arcface.npy   — the 512-d ArcFace recognition embedding (the identity / Stage-2 key)
  2. composed.glb  — MICA shape (identity) wearing SMIRK's expression+pose from this same photo

MICA gives the identity; SMIRK gives the expression/pose; they share the FLAME 2020 basis, so the
identity drops into SMIRK's FLAME and we get a posed mesh of that person. (The composed identity is
currently the RAW MICA shape — a faithful reconstruction; Stage 2 will mutate that shape vector — the
hash — before compose.)

WHY TWO SUBPROCESSES: MICA and SMIRK are separate upstream repos that BOTH ship top-level `utils`,
`configs`, and `datasets` packages, so importing both into one interpreter collides. Rather than do
fragile sys.modules surgery, this driver runs mica_local.py then smirk_local.py as isolated
subprocesses (each with its own clean import state), drops the intermediates in a temp dir, and
assembles only the two final artifacts into <out>. Same venv python is reused, so no env juggling.

This is the thin top of the local stack:  run.py -> mica_local.py + smirk_local.py.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable                                  # reuse this venv's interpreter for both stages
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _stage(title, cmd):
    print(f"\n=== {title} ===", flush=True)           # flush so headers stay ordered vs child output
    subprocess.run(cmd, check=True)                  # inherit stdout/stderr so progress streams live


def run(in_dir, out_dir, device="cpu", detector="antelopev2", keep_work=False):
    in_dir, out_dir = os.path.abspath(in_dir), os.path.abspath(out_dir)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"input folder not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    work = tempfile.mkdtemp(prefix="facehash_")
    mica_out, smirk_out = os.path.join(work, "mica"), os.path.join(work, "smirk")
    try:
        # Stage 1 — MICA: <stem>/{identity.npy, arcface.npy, <stem>.glb}
        _stage("MICA — identity (ArcFace 512-d + FLAME shape 300-d)",
               [PY, os.path.join(HERE, "mica_local.py"), "-i", in_dir, "-o", mica_out,
                "--device", device, "--detector", detector])
        # Stage 2/3 — SMIRK: expression/pose from the photo, composed with MICA's shape -> composed.glb
        _stage("SMIRK — expression/pose + compose (MICA shape + SMIRK pose)",
               [PY, os.path.join(HERE, "smirk_local.py"), "-i", in_dir, "-o", smirk_out,
                "--mica", mica_out, "--device", device])

        # Assemble the clean deliverable: out/<stem>/{arcface.npy, composed.glb} and nothing else
        print(f"\n=== assemble -> {out_dir} ===", flush=True)
        stems = []
        for stem in sorted(os.listdir(mica_out)) if os.path.isdir(mica_out) else []:
            arc = os.path.join(mica_out, stem, "arcface.npy")
            glb = os.path.join(smirk_out, stem, "composed.glb")
            if not os.path.exists(arc):
                continue
            if not os.path.exists(glb):
                print(f"  {stem}: arcface only — SMIRK produced no composed.glb (skipped)")
                continue
            d = os.path.join(out_dir, stem)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(arc, os.path.join(d, "arcface.npy"))
            shutil.copy2(glb, os.path.join(d, "composed.glb"))
            print(f"  {stem}: arcface.npy + composed.glb")
            stems.append(stem)
        return stems
    finally:
        if keep_work:
            print(f"\n(intermediates kept at {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Photos -> per-person {arcface.npy, composed.glb} (MICA identity + SMIRK pose).")
    p.add_argument("-i", "--in_dir", required=True, help="folder of photos")
    p.add_argument("-o", "--out_dir", required=True, help="output folder (writes <stem>/ per photo)")
    p.add_argument("--device", default=os.environ.get("FACE_DEVICE", "cpu"), help="cpu (default) | mps | auto")
    p.add_argument("--detector", default="antelopev2", choices=["antelopev2", "vision"],
                   help="MICA's face detector: antelopev2 (default) | vision (macOS, via loupe)")
    p.add_argument("--keep-work", action="store_true", help="keep the temp dir of intermediates (debug)")
    a = p.parse_args()
    names = run(a.in_dir, a.out_dir, device=a.device, detector=a.detector, keep_work=a.keep_work)
    print(f"\ndone: {len(names)} face(s) -> {os.path.abspath(a.out_dir)}")
