# comfy/workflows

Hand-saved **ComfyUI canvas workflows** (git-tracked), kept separate from `comfy/comfyui/`
(which `to_comfy.py` *generates* from `build_*.py` — don't hand-edit those).

`pod_bootstrap.sh` symlinks ComfyUI's `user/default/workflows/` to this folder, so:

- **Save** in the canvas → writes here → `git add . && git commit && git push`
- a fresh pod's `git pull` (+ the bootstrap symlink) → the workflow is back in your sidebar

So this is where your live-tuned `.json` workflows land and sync. Models and personal photos
are **not** in git — only the workflow recipe.
