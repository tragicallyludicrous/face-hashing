#!/usr/bin/env bash
# setup.sh — one-shot, RE-RUNNABLE setup for the Face-Hashing studio.
#
# Idempotent by design: every step checks for what it needs before fetching/installing, so re-running
# (e.g. after a pod migration on a persistent volume) is fast and does NOT re-download or reinstall.
# In particular it does NOT call SMIRK's quick_install.sh — that script has no guards and re-downloads
# FLAME + EMOCA + a duplicate mica.tar on every run. We fetch only the inference assets, each guarded,
# and share the gated FLAME / mica.tar between MICA and SMIRK so they download AT MOST ONCE.
#
# The two registration-gated weights (FLAME generic_model.pkl, MICA mica.tar) can't be auto-fetched;
# this tells you exactly where to drop them. Tested on macOS arm64; Linux/pod should work.
# Then:  python3 viewer/studio_server.py     (run just the doctor:  python3 viewer/studio_server.py --check)
#
# Override the interpreter:  PYTHON=python3.11 ./setup.sh   (or PYTHON=/path/to/python3.11 ./setup.sh)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"; LOCAL="$ROOT/local"
PYBIN="${PYTHON:-python3.11}"

echo "== Face-Hashing studio setup (re-runnable) =="
command -v "$PYBIN" >/dev/null || { echo "!! need Python 3.11 (set PYTHON=<your python3.11>)"; exit 1; }
"$PYBIN" -c 'import sys;assert sys.version_info[:2]==(3,11),sys.version' \
  || { echo "!! use Python 3.11 — this stack (torch/insightface/chumpy) has no wheels on 3.12+"; exit 1; }

# 1) venv + deps  (skip the whole pip resolve if everything already imports)
VENV="$LOCAL/.venv"
[ -x "$VENV/bin/python" ] || { echo "-- creating $VENV"; "$PYBIN" -m venv "$VENV"; }
PY="$VENV/bin/python"
DEP_PROBE='import torch,torchvision,insightface,onnxruntime,trimesh,yacs,loguru,cv2,skimage,numpy,gdown,tqdm,face_alignment,mediapipe,matplotlib,chumpy'
if "$PY" -c "$DEP_PROBE" 2>/dev/null; then
  echo "-- deps already present, skipping pip"
else
  echo "-- installing deps (one or more missing)"
  "$PY" -m pip install -U pip setuptools wheel
  "$PY" -m pip install torch torchvision insightface onnxruntime \
        trimesh loguru yacs opencv-python scikit-image numpy gdown tqdm face-alignment mediapipe matplotlib
  "$PY" -m pip install --no-build-isolation chumpy   # setup.py imports pip -> no build isolation
fi

# 2) MICA checkout + Mac/CPU patch (idempotent; the name is historical, fine on Linux)
[ -d "$LOCAL/MICA" ] || { echo "-- cloning MICA"; git clone https://github.com/Zielon/MICA.git "$LOCAL/MICA"; }
"$PY" "$LOCAL/patch_mica_for_mac.py" "$LOCAL/MICA"

# 3) SMIRK checkout + inference assets.  We deliberately do NOT run SMIRK's quick_install.sh (it is
#    interactive and re-downloads everything every time). Renderer-free inference needs only these three;
#    the eyelid / landmark / template assets already ship in the git clone.
[ -d "$LOCAL/smirk" ] || { echo "-- cloning SMIRK"; git clone https://github.com/georgeretsi/smirk.git "$LOCAL/smirk"; }

SMIRK_PT="$LOCAL/smirk/pretrained_models/SMIRK_em1.pt"
if [ ! -f "$SMIRK_PT" ]; then
  echo "-- fetching SMIRK_em1.pt"
  mkdir -p "$LOCAL/smirk/pretrained_models"
  "$PY" -m gdown --id 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O "$SMIRK_PT" \
    || echo "  (gdown hiccup — get SMIRK_em1.pt from SMIRK's repo into $SMIRK_PT)"
fi

LMARK="$LOCAL/smirk/assets/face_landmarker.task"
if [ ! -f "$LMARK" ]; then
  echo "-- fetching mediapipe face_landmarker"
  mkdir -p "$LOCAL/smirk/assets"
  wget -q -O "$LMARK" \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    || echo "  (face_landmarker fetch failed — mediapipe will still fall back to RetinaFace)"
fi

# 3b) antelopev2 detector pack (guarded — only fetches if no .onnx present)
"$PY" - <<'PY' || echo "  (antelopev2 will auto-fetch on the first pipeline run)"
import os, glob, shutil
from insightface.utils.storage import ensure_available
base = os.path.expanduser('~/.insightface/models/antelopev2')
if glob.glob(base + '/**/*.onnx', recursive=True):
    print('antelopev2: present, skipping')
else:
    ensure_available('models', 'antelopev2', root=os.path.expanduser('~/.insightface'))
    os.makedirs(base, exist_ok=True)
    for f in glob.glob(base + '/**/*.onnx', recursive=True):
        dst = os.path.join(base, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst): shutil.move(f, dst)
    print('antelopev2:', sorted(os.listdir(base)))
PY

# 4) FLAME 2020 (gated, non-commercial): download ONCE, share between MICA and SMIRK.
#    Look in both consumers (and anywhere under local/) first; copy into whichever is missing.
#    Only print the manual instructions if it's absent everywhere — never auto-re-download.
mkdir -p "$LOCAL/MICA/data/FLAME2020" "$LOCAL/smirk/assets/FLAME2020"
MICA_FLAME="$LOCAL/MICA/data/FLAME2020/generic_model.pkl"
SMIRK_FLAME="$LOCAL/smirk/assets/FLAME2020/generic_model.pkl"
if [ -f "$MICA_FLAME" ]; then FOUND_FLAME="$MICA_FLAME"
elif [ -f "$SMIRK_FLAME" ]; then FOUND_FLAME="$SMIRK_FLAME"
else FOUND_FLAME="$(find "$LOCAL" -name generic_model.pkl 2>/dev/null | head -1)"; fi
if [ -n "${FOUND_FLAME:-}" ] && [ -f "$FOUND_FLAME" ]; then
  [ -f "$MICA_FLAME" ]  || { echo "-- sharing FLAME -> MICA";  cp "$FOUND_FLAME" "$MICA_FLAME"; }
  [ -f "$SMIRK_FLAME" ] || { echo "-- sharing FLAME -> SMIRK"; cp "$FOUND_FLAME" "$SMIRK_FLAME"; }
else
  echo "!! FLAME generic_model.pkl not found (registration-gated, non-commercial)."
  echo "   Get FLAME2020.zip from https://flame.is.tue.mpg.de, unzip, and place generic_model.pkl at:"
  echo "     $MICA_FLAME"
  echo "   Re-run setup.sh — it will share that copy with SMIRK and never re-download it."
fi

# 4b) MICA checkpoint (gated): reuse SMIRK's bundled mica.tar if a previous setup left one.
mkdir -p "$LOCAL/MICA/data/pretrained"
MICA_TAR="$LOCAL/MICA/data/pretrained/mica.tar"
if [ ! -f "$MICA_TAR" ]; then
  if [ -f "$LOCAL/smirk/assets/mica.tar" ]; then
    echo "-- reusing SMIRK's mica.tar -> MICA"; cp "$LOCAL/smirk/assets/mica.tar" "$MICA_TAR"
  else
    echo "!! MICA mica.tar not found (research-only) — download per https://github.com/Zielon/MICA into:"
    echo "     $MICA_TAR"
  fi
fi

# 5) FLAME slider basis for the viewer (guarded — needs the pkl, built once)
if [ -f "$MICA_FLAME" ] && [ ! -f "$ROOT/viewer/flame/flame_basis.bin" ]; then
  echo "-- exporting FLAME slider basis"
  "$PY" "$ROOT/tools/export_flame_basis.py" || echo "  (basis export failed — check the FLAME pkl)"
fi

echo; echo "== setup check =="
"$PY" "$ROOT/viewer/studio_server.py" --check || true
cat <<'EOF'

Re-runnable: nothing above re-downloads what you already have. Any [XX] is almost certainly one of the
two gated weights (placed once, then shared/reused forever):
  • FLAME generic_model.pkl  -> register at https://flame.is.tue.mpg.de, drop in local/MICA/data/FLAME2020/
  • MICA mica.tar            -> download per https://github.com/Zielon/MICA into local/MICA/data/pretrained/
Re-check anytime (no downloads):  python3 viewer/studio_server.py --check
When all green:                   python3 viewer/studio_server.py
EOF
