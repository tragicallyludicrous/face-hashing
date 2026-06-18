# Quickstart — the drag-drop face studio

Clone, set up, and play: drop a JPEG in the browser → get a 3-D face you can twist, randomize, and
"hash" (same photo + same key → same new face).

```bash
git clone git@github.com:tragicallyludicrous/face-hashing.git
cd face-hashing
./setup.sh                                   # PYTHON=python3.11 ./setup.sh  if needed
python3 viewer/studio_server.py              # opens the studio; drag a JPEG onto it
```

`./setup.sh` builds the venv, clones MICA + SMIRK, fetches the detector, and exports the slider basis,
then prints a setup check. Re-run the check anytime:

```bash
python3 viewer/studio_server.py --check      # lists every prerequisite as [OK]/[XX] with a fix
```

## The two things setup can't fetch for you (registration-gated)

When `--check` shows `[XX]`, it's almost always one of these — grab them once, drop them in, re-check:

| Missing | Where | Put it |
|---|---|---|
| `FLAME 2020 generic_model.pkl` | register at <https://flame.is.tue.mpg.de> | `local/MICA/data/FLAME2020/` |
| `MICA mica.tar` | <https://github.com/Zielon/MICA> (research-only) | `local/MICA/data/pretrained/` |

(SMIRK's `quick_install.sh` often pulls a FLAME copy too — `setup.sh` reuses it for MICA automatically,
so FLAME may already be satisfied.)

## Requirements

- **Python 3.11** (arm64 on Apple Silicon). Not 3.12+ — this stack has no wheels there.
- ~2 GB of model weights; first face build takes ~10–60 s (CPU). No GPU required.
- macOS arm64 is the tested path; Linux should work (the pipeline is CPU / renderer-free).

## Once it's running

See **`viewer/STUDIO.md`** for the studio itself (drag-drop, sliders, Hash/Randomize, LAN sharing).
Full pipeline internals: **`CLAUDE.md`** and **`local/README.md`**.
