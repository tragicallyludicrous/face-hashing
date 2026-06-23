#!/usr/bin/env python3
"""
studio_server.py — drag-drop Face-Hashing studio: serves the viewer AND runs the pipeline on upload.

    # from the repo ROOT (uses local/.venv automatically if present):
    python3 viewer/studio_server.py
    # then open the printed URL and DRAG A JPEG onto the studio.

It serves the repo root statically (so /viewer, /local/out, /local/in, /viewer/flame resolve, with the
directory listings the studio parses) and adds one endpoint:

    POST /api/process   body = raw image bytes, header X-Filename = original name
        -> saves the photo, runs `local/run.py` on just that photo, and returns {"stem": ...}
           once out/<stem>/ has the meshes. The page then loads that folder via its normal path.

Localhost-only by default (it executes the local pipeline on whatever you drop). Use --host 0.0.0.0 to
expose on the LAN only if you trust the network. Each upload spawns run.py fresh (~1-2 min/photo).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # viewer/.. == repo root
LOCAL = os.path.join(REPO, "local")
RUN = os.path.join(LOCAL, "run.py")
OUT = os.path.join(LOCAL, "out")
IN = os.path.join(LOCAL, "in")
VENV = os.path.join(LOCAL, ".venv", "bin", "python")
PY = os.environ.get("FACEHASH_PY") or (VENV if os.path.exists(VENV) else sys.executable)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_BYTES = 40 * 1024 * 1024


def safe_stem(name):
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "face"
    return f"{base[:40]}-{uuid.uuid4().hex[:6]}"


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/process":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BYTES:
                return self._json(400, {"error": "empty or too-large upload (40MB cap)"})
            data = self.rfile.read(n)
            fname = self.headers.get("X-Filename", "upload.jpg")
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                ext = ".jpg"
            stem = safe_stem(fname)
            os.makedirs(IN, exist_ok=True)
            os.makedirs(OUT, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="studio_up_") as tmp:
                src = os.path.join(tmp, stem + ext)           # one-file dir -> run.py does just this photo
                with open(src, "wb") as f:
                    f.write(data)
                shutil.copy2(src, os.path.join(IN, stem + ext))   # so the studio thumbnail can show it
                print(f"[studio] processing {stem}{ext} ({n} bytes) with {PY} …", flush=True)
                t0 = time.time()
                r = subprocess.run([PY, RUN, "-i", tmp, "-o", OUT],
                                   cwd=LOCAL, capture_output=True, text=True, env={**os.environ})
                dt = time.time() - t0
            if r.returncode != 0:
                sys.stdout.write((r.stdout or "")[-2000:] + "\n" + (r.stderr or "")[-2000:] + "\n")
                return self._json(500, {"error": "pipeline failed", "detail": (r.stderr or r.stdout or "")[-600:]})
            comp = os.path.join(OUT, stem, f"{stem}_composed.glb")
            if not os.path.exists(comp):
                return self._json(422, {"error": "no face found / no mesh produced",
                                        "detail": (r.stdout or "")[-600:]})
            print(f"[studio] done {stem} in {dt:.0f}s", flush=True)
            return self._json(200, {"stem": stem, "seconds": round(dt)})
        except Exception as e:                                # noqa: BLE001 — report any failure to the page
            return self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):                        # quieter: skip the per-asset GET spam
        if self.command == "POST" or "api" in (self.path or ""):
            super().log_message(fmt, *args)


def check_setup():
    """Report each prerequisite as OK/XX with a one-line fix. Exit 0 if ready, else 1."""
    M = os.path.join(LOCAL, "MICA")
    S = os.path.join(LOCAL, "smirk")
    antelope = os.path.expanduser("~/.insightface/models/antelopev2")
    venv_py = os.environ.get("FACEHASH_PY") or VENV
    items = [
        ("Python venv", os.path.exists(venv_py),
         "./setup.sh   (external/reuse venv? set FACEHASH_PY=/path/to/venv/bin/python)"),
        ("MICA checkout", os.path.isdir(M),
         "git clone https://github.com/Zielon/MICA.git local/MICA && python local/patch_mica_for_mac.py local/MICA"),
        ("FLAME 2020 generic_model.pkl", os.path.exists(os.path.join(M, "data/FLAME2020/generic_model.pkl")),
         "registration-gated — get it at https://flame.is.tue.mpg.de, put in local/MICA/data/FLAME2020/"),
        ("MICA landmark_embedding.npy", os.path.exists(os.path.join(M, "data/FLAME2020/landmark_embedding.npy")),
         "ships with the MICA repo clone"),
        ("MICA checkpoint mica.tar", os.path.exists(os.path.join(M, "data/pretrained/mica.tar")),
         "research-only — download per MICA's repo into local/MICA/data/pretrained/"),
        ("SMIRK checkout", os.path.isdir(S),
         "git clone https://github.com/georgeretsi/smirk.git local/smirk"),
        ("SMIRK weights SMIRK_em1.pt", os.path.exists(os.path.join(S, "pretrained_models/SMIRK_em1.pt")),
         "./setup.sh  (or: local/.venv/bin/python -m gdown "
         "'https://drive.google.com/uc?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE' "
         "-O local/smirk/pretrained_models/SMIRK_em1.pt)"),
        ("antelopev2 detector", bool(glob.glob(os.path.join(antelope, "*.onnx"))),
         "auto-fetches on first pipeline run (or local/README.md §3b)"),
        ("FLAME slider basis (viewer/flame)",
         os.path.exists(os.path.join(REPO, "viewer/flame/flame_basis.json"))
         and os.path.exists(os.path.join(REPO, "viewer/flame/flame_basis.bin")),
         "local/.venv/bin/python tools/export_flame_basis.py  (needs the FLAME pkl)"),
    ]
    missing_mods = []
    if os.path.exists(venv_py):
        mods = ["numpy", "torch", "cv2", "insightface", "onnxruntime", "skimage", "trimesh", "chumpy", "mediapipe", "timm"]
        code = "import importlib.util as u;print(' '.join(m for m in %r if u.find_spec(m) is None))" % mods
        try:
            missing_mods = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=90).stdout.split()
        except Exception as e:                                # noqa: BLE001
            missing_mods = [f"<probe failed: {e}>"]
    print("\nFace-Hashing studio — setup check\n" + "=" * 42)
    ok_all = True
    for label, ok, fix in items:
        print(f"  [{'OK' if ok else 'XX'}] {label}")
        if not ok:
            ok_all = False
            print(f"         -> {fix}")
    if os.path.exists(venv_py):
        if missing_mods:
            ok_all = False
            print(f"  [XX] venv packages missing: {', '.join(missing_mods)}\n         -> ./setup.sh")
        else:
            print("  [OK] venv packages (torch, insightface, mediapipe, …)")
    print("=" * 42)
    print("READY — run:  python3 viewer/studio_server.py" if ok_all
          else "NOT READY — fix the [XX] items above (./setup.sh, then re-check).")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description="Drag-drop Face-Hashing studio server.")
    ap.add_argument("--check", action="store_true", help="check prerequisites and exit")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default localhost; 0.0.0.0 = LAN)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    a = ap.parse_args()
    if a.check:
        sys.exit(check_setup())
    if not os.path.exists(RUN):
        sys.exit(f"run.py not found at {RUN} — launch from the repo root.")
    if not os.path.exists(os.path.join(REPO, "viewer/flame/flame_basis.bin")):
        print("  ! viewer/flame basis missing — sliders won't load. Run:  python3 viewer/studio_server.py --check")
    httpd = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=REPO))
    url = f"http://{'localhost' if a.host=='127.0.0.1' else a.host}:{a.port}/viewer/studio.html"
    print(f"\n  Face-Hashing studio:  {url}\n  drag a JPEG onto the page to build a face")
    print(f"  serving: {REPO}\n  pipeline python: {PY}\n  (Ctrl-C to stop)\n")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
