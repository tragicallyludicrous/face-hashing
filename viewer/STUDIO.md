# Face-Hashing studio — drag & drop

Turn a photo into a 3-D face you can twist, randomize, and "hash" — in the browser.

## Run it (one command)

```bash
python3 viewer/studio_server.py      # from the repo root
```

A browser tab opens. **Drag a JPEG anywhere onto the page** → wait ~10–60 s → your face appears as
two meshes (identity + posed), with 300 sliders, a **Hash** button (same photo + same key → same new
face), and **Randomize**.

That's it. `Ctrl-C` in the terminal to stop. Share it on your LAN with `--port 8080 --host 0.0.0.0`
(only on a network you trust — it runs the pipeline on whatever gets dropped).

## What it needs (one-time setup on the machine that runs it)

This drives the real pipeline, so that machine needs the project set up:
- `local/.venv` with MICA + SMIRK installed (see `local/README.md`) — the server auto-detects it.
- The MICA/SMIRK model weights downloaded per their licenses.
- `viewer/flame/` present (run `tools/export_flame_basis.py`) — the slider basis.

It can't be a zero-setup download: the FLAME weights are registration-gated and the models are large.
The *studio* is just two files (`viewer/studio.html` + `viewer/studio_server.py`); everything heavy is
the existing local stack.

## If the drop says "could not build"

- **"is studio_server.py running?"** — you opened `studio.html` with a plain file server. Use the
  command above instead (the drop needs the server's `/api/process`).
- **"no face found"** — try a clearer, front-facing photo.
- Watch the terminal — the pipeline streams its errors there.
