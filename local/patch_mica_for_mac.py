#!/usr/bin/env python3
"""
patch_mica_for_mac.py — make a MICA checkout run its demo on macOS (CPU or Apple MPS),
with no CUDA and no PyTorch3D.

WHY THIS IS ENOUGH (verified by reading MICA's source, 2026-06-01):
MICA's entire *inference* import graph is PyTorch3D-free —

    demo.py -> micalib/models/mica.py -> micalib/base_model.py
            -> models/{arcface,generator,flame}.py -> utils/masking.py

PyTorch3D appears only in MICA's training/validation/rendering modules, which the demo never
imports. So porting to a GPU-less Mac is NOT "stub the renderer"; it is just:

  1. chumpy shim   — FLAME's generic_model.pkl unpickles chumpy objects; chumpy is unmaintained
                     (calls inspect.getargspec, gone in 3.11+, and `from numpy import bool,...`,
                     gone in numpy 2.x). Same shim we use in Colab, prepended on disk.
  2. face_alignment enum — MICA's utils/landmark_detector.py uses the old LandmarksType._2D name.
  3. demo.py CPU-safety — it hardcodes device='cuda:0', tensor.cuda(), and torch.load(args.m)
                     with no map_location (fails CPU-side) / no weights_only (torch>=2.6 default
                     flipped to True and the checkpoint carries non-tensor payload). We also
                     neutralize torch.cuda.amp.autocast(), which arcface.py uses, on a CUDA-less box.

Device is chosen at runtime by demo.py from $MICA_DEVICE (default: mps if available else cpu).
For the FIRST verification run, force CPU — fewest surprises:  MICA_DEVICE=cpu python demo.py ...

Idempotent. Run once against your checkout:

    python local/patch_mica_for_mac.py /path/to/MICA
"""
import importlib.util
import os
import re
import sys

SHIM_TAG = "on-py312 shim"
DEMO_TAG = "MICA-on-Mac shim"


def patch_chumpy():
    """Prepend the inspect/numpy-alias shim to chumpy/__init__.py (locate WITHOUT importing it)."""
    spec = importlib.util.find_spec("chumpy")
    if spec is None or not spec.origin:
        print("  chumpy: NOT installed yet — `pip install chumpy`, then re-run this script")
        return
    shim = (
        "import numpy as _np, inspect as _inspect\n"
        "if not hasattr(_inspect, 'getargspec'): _inspect.getargspec = _inspect.getfullargspec\n"
        "for _a, _r in [('bool',bool),('int',int),('float',float),('complex',complex),"
        "('object',object),('str',str),('unicode',str)]:\n"
        "    if not hasattr(_np, _a): setattr(_np, _a, _r)\n"
        f"# --- DECA/MICA-{SHIM_TAG} ---\n"
    )
    s = open(spec.origin).read()
    if SHIM_TAG in s:
        print("  chumpy: already shimmed")
        return
    open(spec.origin, "w").write(shim + s)
    print("  chumpy: shim prepended ->", spec.origin)


def patch_landmark_detector(mica):
    f = os.path.join(mica, "utils", "landmark_detector.py")
    if not os.path.exists(f):
        print("  landmark_detector.py: not found (skipping)")
        return
    s = orig = open(f).read()
    s = s.replace("LandmarksType._2D", "LandmarksType.TWO_D")    # face_alignment enum rename
    # No CUDA on a Mac: drop the hardcoded CUDA-only provider (onnxruntime CPU on Apple Silicon).
    s = s.replace("['CUDAExecutionProvider']", "['CPUExecutionProvider']")
    if s != orig:
        open(f, "w").write(s)
        print("  landmark_detector.py: _2D->TWO_D + CUDA->CPU provider")
    else:
        print("  landmark_detector.py: already patched")


def patch_numpy_removed_aliases(mica):
    """numpy 2.0 removed np.Inf / np.NaN / np.NAN / np.infty; MICA's util code still uses them.

    The chumpy shim restores some aliases on the `numpy` module, but MICA's own source references
    these names directly (e.g. datasets/creation/util.py:get_center -> np.Inf), so rewrite them.
    """
    repls = [(r"\bnp\.Inf\b", "np.inf"), (r"\bnp\.NaN\b", "np.nan"),
             (r"\bnp\.NAN\b", "np.nan"), (r"\bnp\.infty\b", "np.inf")]
    changed = 0
    for root, _, files in os.walk(mica):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            s = orig = open(p).read()
            for pat, rep in repls:
                s = re.sub(pat, rep, s)
            if s != orig:
                open(p, "w").write(s)
                changed += 1
    print(f"  numpy aliases: rewrote np.Inf/NaN/NAN/infty in {changed} file(s)")


def patch_demo(mica):
    f = os.path.join(mica, "demo.py")
    s = orig = open(f).read()

    # (1) (re)inject the top-of-file shim. Strip any prior copy first so re-running UPDATES it
    #     (idempotent AND upgradeable). The shim: MPS fallback env, the runtime device, a
    #     CUDA-autocast no-op, and — crucially — a GLOBAL torch.load override forcing
    #     map_location='cpu', weights_only=False. The global override is what makes MICA's own
    #     load_model() (micalib/models/mica.py) CPU-safe, not just demo.py's load_checkpoint.
    block = re.compile(rf"# --- {re.escape(DEMO_TAG)} .*?# --- end {re.escape(DEMO_TAG)} ---\n", re.S)
    s = block.sub("", s)
    preamble = (
        f"# --- {DEMO_TAG} (injected by patch_mica_for_mac.py) ---\n"
        "import os as _os\n"
        "_os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')\n"
        "import contextlib as _cl, torch as _torch\n"
        "_DEVICE = _os.environ.get('MICA_DEVICE') or "
        "('mps' if _torch.backends.mps.is_available() else 'cpu')\n"
        "_torch.cuda.amp.autocast = lambda *a, **k: _cl.nullcontext()\n"
        "_mica_orig_load = _torch.load\n"
        "def _mica_cpu_load(*a, **k):\n"
        "    k.setdefault('map_location', 'cpu'); k.setdefault('weights_only', False)\n"
        "    return _mica_orig_load(*a, **k)\n"
        "_torch.load = _mica_cpu_load\n"
        f"# --- end {DEMO_TAG} ---\n"
    )
    s = preamble + s

    # (2) device string + tensor placement -> the runtime-chosen _DEVICE
    s = s.replace("'cuda:0'", "_DEVICE").replace('"cuda:0"', "_DEVICE")
    s = s.replace(".cuda()", ".to(_DEVICE)")

    # (3) demo's own checkpoint load (redundant with the global override above, kept as belt-and-suspenders)
    s = re.sub(r"torch\.load\(\s*args\.m\s*\)",
               "torch.load(args.m, map_location='cpu', weights_only=False)", s)

    # (4) MPS has no float64; to_batch builds these from float64 numpy (imread()/255.). Cast to
    #     float32 before the device move — a no-op on CPU, required on MPS.
    s = s.replace("torch.tensor(image).to(_DEVICE)", "torch.tensor(image).float().to(_DEVICE)")
    s = s.replace("torch.tensor(arcface).to(_DEVICE)", "torch.tensor(arcface).float().to(_DEVICE)")

    if s != orig:
        open(f, "w").write(s)
        print("  demo.py: CPU/MPS-patched (device, .cuda(), GLOBAL torch.load, autocast)")
    else:
        print("  demo.py: already patched")


if __name__ == "__main__":
    mica = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "MICA")
    if not os.path.isdir(mica):
        sys.exit(f"not a MICA checkout: {mica}\nusage: python local/patch_mica_for_mac.py /path/to/MICA")
    print("patching MICA for macOS at", mica)
    patch_chumpy()
    patch_landmark_detector(mica)
    patch_numpy_removed_aliases(mica)
    patch_demo(mica)
    print("done.")
