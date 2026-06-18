#!/usr/bin/env bash
# setup.sh — one-shot setup for the Face-Hashing drag-drop studio.
#
# Automates the local/README runbook: venv + deps, MICA + SMIRK checkouts, antelopev2, FLAME basis.
# The two registration-gated weights (FLAME generic_model.pkl, MICA mica.tar) can't be auto-fetched;
# this tells you exactly where to drop them. Re-runnable. Tested on macOS arm64; Linux should work
# (the pipeline is CPU / renderer-free). Then:  python3 viewer/studio_server.py
#
# Override the interpreter:  PYTHON=python3.11 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"; LOCAL="$ROOT/local"
PYBIN="${PYTHON:-python3.11}"

echo "== Face-Hashing studio setup =="
command -v "$PYBIN" >/dev/null || { echo "!! need Python 3.11 (set PYTHON=<your python3.11>)"; exit 1; }
"$PYBIN" -c 'import sys;assert sys.version_info[:2]==(3,11),sys.version' \
  || { echo "!! use Python 3.11 — this stack (torch/insightface/chumpy) has no wheels on 3.12+"; exit 1; }

# 1) venv + deps
VENV="$LOCAL/.venv"
[ -x "$VENV/bin/python" ] || { echo "-- creating $VENV"; "$PYBIN" -m venv "$VENV"; }
PY="$VENV/bin/python"
"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install torch torchvision insightface onnxruntime \
      trimesh loguru yacs opencv-python scikit-image numpy gdown tqdm face-alignment mediapipe matplotlib
"$PY" -m pip install --no-build-isolation chumpy   # setup.py imports pip -> no build isolation

# 2) MICA checkout + Mac/CPU patch (idempotent; the name is historical, fine on Linux)
[ -d "$LOCAL/MICA" ] || { echo "-- cloning MICA"; git clone https://github.com/Zielon/MICA.git "$LOCAL/MICA"; }
"$PY" "$LOCAL/patch_mica_for_mac.py" "$LOCAL/MICA"

# 3) SMIRK checkout + assets (SMIRK_em1.pt + FLAME/eyelid/mediapipe)
[ -d "$LOCAL/smirk" ] || { echo "-- cloning SMIRK"; git clone https://github.com/georgeretsi/smirk.git "$LOCAL/smirk"; }
[ -f "$LOCAL/smirk/pretrained_models/SMIRK_em1.pt" ] || \
  { echo "-- SMIRK quick_install"; ( cd "$LOCAL/smirk" && bash quick_install.sh ) || echo "  (quick_install hiccup — see local/smirk/quick_install.sh)"; }

# 3b) antelopev2 detector pack (the README §3b flatten)
"$PY" - <<'PY' || echo "  (antelopev2 will auto-fetch on the first pipeline run)"
import os, glob, shutil
from insightface.utils.storage import ensure_available
base = os.path.expanduser('~/.insightface/models/antelopev2')
if not glob.glob(base + '/**/*.onnx', recursive=True):
    ensure_available('models', 'antelopev2', root=os.path.expanduser('~/.insightface'))
os.makedirs(base, exist_ok=True)
for f in glob.glob(base + '/**/*.onnx', recursive=True):
    dst = os.path.join(base, os.path.basename(f))
    if os.path.abspath(f) != os.path.abspath(dst): shutil.move(f, dst)
print('antelopev2:', sorted(os.listdir(base)))
PY

# 4) FLAME (gated): reuse SMIRK's copy for MICA if quick_install fetched one
mkdir -p "$LOCAL/MICA/data/FLAME2020" "$LOCAL/MICA/data/pretrained"
MICA_FLAME="$LOCAL/MICA/data/FLAME2020/generic_model.pkl"
if [ ! -f "$MICA_FLAME" ]; then
  FOUND="$(find "$LOCAL/smirk/assets" -name generic_model.pkl 2>/dev/null | head -1 || true)"
  [ -n "$FOUND" ] && { echo "-- reusing SMIRK's FLAME for MICA"; cp "$FOUND" "$MICA_FLAME"; }
fi

# 5) FLAME slider basis for the viewer (needs the pkl)
if [ -f "$MICA_FLAME" ] && [ ! -f "$ROOT/viewer/flame/flame_basis.bin" ]; then
  echo "-- exporting FLAME slider basis"
  "$PY" "$ROOT/tools/export_flame_basis.py" || echo "  (basis export failed — check the FLAME pkl)"
fi

echo; echo "== setup check =="
"$PY" "$ROOT/viewer/studio_server.py" --check || true
cat <<'EOF'

Any [XX] above is almost certainly one of the two gated weights:
  • FLAME generic_model.pkl  -> register at https://flame.is.tue.mpg.de, drop in local/MICA/data/FLAME2020/
  • MICA mica.tar            -> download per https://github.com/Zielon/MICA into local/MICA/data/pretrained/
Fix those, re-run:  python3 viewer/studio_server.py --check
When all green:     python3 viewer/studio_server.py
EOF
